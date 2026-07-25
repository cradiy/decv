use std::{
    fmt,
    io::{self, Read},
    ops::Range,
};

use decv_core::MediaInput;
use ureq::{
    Agent,
    http::{
        HeaderMap, HeaderName, HeaderValue, StatusCode,
        header::{
            ACCEPT_ENCODING, AUTHORIZATION, CONTENT_ENCODING, CONTENT_RANGE, COOKIE, ETAG,
            IF_RANGE, LAST_MODIFIED, PROXY_AUTHORIZATION, RANGE,
        },
    },
};

use crate::{CachedRangeInput, RangeCacheConfig, RangeCacheStats, RangeFetcher, RangeInputStats};

/// Builder for a strict, cached HTTP byte-range media input.
///
/// Construction performs one `bytes=0-0` request to discover and validate the
/// stable object length. Servers that ignore ranges are rejected.
pub struct HttpRangeInputBuilder {
    url: String,
    agent: Agent,
    headers: HeaderMap,
    cache_config: RangeCacheConfig,
}

impl HttpRangeInputBuilder {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            agent: Agent::new_with_defaults(),
            headers: HeaderMap::new(),
            cache_config: RangeCacheConfig::default(),
        }
    }

    /// Uses the supplied agent and its shared connection pool and settings.
    pub fn agent(mut self, agent: Agent) -> Self {
        self.agent = agent;
        self
    }

    /// Adds a header to every metadata and range request.
    ///
    /// `Range`, `If-Range`, and `Accept-Encoding` are controlled internally so
    /// response validation cannot be bypassed.
    pub fn header(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> io::Result<Self> {
        let name = HeaderName::from_bytes(name.as_ref().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid HTTP header name"))?;
        let value = HeaderValue::from_str(value.as_ref()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "invalid HTTP header value")
        })?;
        reject_managed_header(&name)?;
        self.headers.append(name, value);
        Ok(self)
    }

    pub fn cache_config(mut self, cache_config: RangeCacheConfig) -> Self {
        self.cache_config = cache_config;
        self
    }

    pub fn build(self) -> io::Result<HttpRangeInput> {
        let fetcher = HttpRangeFetcher::probe(self.url, self.agent, self.headers)?;
        let input = CachedRangeInput::new(fetcher, self.cache_config)?;
        Ok(HttpRangeInput { input })
    }
}

impl fmt::Debug for HttpRangeInputBuilder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HttpRangeInputBuilder")
            .field("endpoint", &RedactedEndpoint)
            .field("headers", &RedactedHeaders(&self.headers))
            .field("cache_config", &self.cache_config)
            .finish_non_exhaustive()
    }
}

/// A known-length HTTP object exposed through [`MediaInput`].
///
/// Reads are synchronous. Callers should perform demuxing and explicit
/// prefetching on worker threads.
pub struct HttpRangeInput {
    input: CachedRangeInput<HttpRangeFetcher>,
}

impl HttpRangeInput {
    pub fn open(url: impl Into<String>) -> io::Result<Self> {
        Self::builder(url).build()
    }

    pub fn builder(url: impl Into<String>) -> HttpRangeInputBuilder {
        HttpRangeInputBuilder::new(url)
    }

    pub const fn content_length(&self) -> u64 {
        self.input.content_length()
    }

    pub const fn cache_config(&self) -> RangeCacheConfig {
        self.input.config()
    }

    pub const fn stats(&self) -> &RangeCacheStats {
        self.input.stats()
    }

    pub fn stats_snapshot(&self) -> RangeInputStats {
        self.stats().snapshot()
    }

    pub fn prefetch(&self, offset: u64, length: usize) -> io::Result<()> {
        self.input.prefetch(offset, length)
    }

    pub fn clear_cache(&self) -> io::Result<()> {
        self.input.clear()
    }
}

impl fmt::Debug for HttpRangeInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HttpRangeInput")
            .field("endpoint", &RedactedEndpoint)
            .field("content_length", &self.content_length())
            .field("cache_config", &self.cache_config())
            .field("stats", &self.stats_snapshot())
            .finish()
    }
}

impl MediaInput for HttpRangeInput {
    fn len(&self) -> io::Result<Option<u64>> {
        MediaInput::len(&self.input)
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
        self.input.read_at(offset, buffer)
    }
}

struct HttpRangeFetcher {
    url: String,
    agent: Agent,
    headers: HeaderMap,
    content_length: u64,
    validator: Option<Validator>,
}

impl HttpRangeFetcher {
    fn probe(url: String, agent: Agent, headers: HeaderMap) -> io::Result<Self> {
        let mut response = send_request(&agent, &url, &headers, 0..1, None)?;
        validate_content_encoding(response.headers())?;
        require_partial_content(response.status(), false)?;
        let content_range = parse_response_content_range(response.headers())?;
        if content_range.start != 0 || content_range.end != 1 {
            return Err(invalid_response(
                "metadata response did not contain the requested byte",
            ));
        }
        if content_range.total == 0 {
            return Err(invalid_response("remote media object is empty"));
        }
        let validator = response_validator(response.headers());
        read_exact_response(&mut response, 1)?;

        Ok(Self {
            url,
            agent,
            headers,
            content_length: content_range.total,
            validator,
        })
    }

    fn validate_version(&self, headers: &HeaderMap) -> io::Result<()> {
        let Some(expected) = &self.validator else {
            return Ok(());
        };
        let Some(actual) = headers.get(&expected.name) else {
            return Ok(());
        };
        if actual != expected.value {
            return Err(invalid_response(
                "remote media object changed while reading ranges",
            ));
        }
        Ok(())
    }
}

impl fmt::Debug for HttpRangeFetcher {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HttpRangeFetcher")
            .field("endpoint", &RedactedEndpoint)
            .field("headers", &RedactedHeaders(&self.headers))
            .field("content_length", &self.content_length)
            .field("has_validator", &self.validator.is_some())
            .finish_non_exhaustive()
    }
}

impl RangeFetcher for HttpRangeFetcher {
    fn len(&self) -> io::Result<u64> {
        Ok(self.content_length)
    }

    fn fetch_range(&self, range: Range<u64>) -> io::Result<Vec<u8>> {
        if range.start >= range.end || range.end > self.content_length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "HTTP byte range is outside the known media length",
            ));
        }

        let mut response = send_request(
            &self.agent,
            &self.url,
            &self.headers,
            range.clone(),
            self.validator.as_ref(),
        )?;
        validate_content_encoding(response.headers())?;
        require_partial_content(response.status(), self.validator.is_some())?;
        let returned = parse_response_content_range(response.headers())?;
        if returned.start != range.start
            || returned.end != range.end
            || returned.total != self.content_length
        {
            return Err(invalid_response(
                "server returned bytes for a different range or object length",
            ));
        }
        self.validate_version(response.headers())?;
        let expected = usize::try_from(range.end - range.start)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "HTTP range is too large"))?;
        read_exact_response(&mut response, expected)
    }
}

#[derive(Clone)]
struct Validator {
    name: HeaderName,
    value: HeaderValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParsedContentRange {
    start: u64,
    end: u64,
    total: u64,
}

fn send_request(
    agent: &Agent,
    url: &str,
    headers: &HeaderMap,
    range: Range<u64>,
    validator: Option<&Validator>,
) -> io::Result<ureq::http::Response<ureq::Body>> {
    let inclusive_end = range
        .end
        .checked_sub(1)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "HTTP range is empty"))?;
    let mut request = agent
        .get(url)
        .header(ACCEPT_ENCODING, HeaderValue::from_static("identity"))
        .header(RANGE, format!("bytes={}-{}", range.start, inclusive_end));
    for (name, value) in headers {
        request = request.header(name.clone(), value.clone());
    }
    if let Some(validator) = validator {
        request = request.header(IF_RANGE, validator.value.clone());
    }
    request.call().map_err(|error| {
        let kind = error.into_io().kind();
        io::Error::new(kind, "HTTP range request failed")
    })
}

fn require_partial_content(status: StatusCode, conditional: bool) -> io::Result<()> {
    if status == StatusCode::PARTIAL_CONTENT {
        return Ok(());
    }
    let message = if conditional && status == StatusCode::OK {
        "remote media object changed while reading ranges"
    } else {
        "server does not provide a valid HTTP partial-content response"
    };
    Err(invalid_response(message))
}

fn validate_content_encoding(headers: &HeaderMap) -> io::Result<()> {
    let Some(value) = headers.get(CONTENT_ENCODING) else {
        return Ok(());
    };
    let encoding = value
        .to_str()
        .map_err(|_| invalid_response("response contains an invalid Content-Encoding header"))?;
    if encoding.eq_ignore_ascii_case("identity") {
        Ok(())
    } else {
        Err(invalid_response(
            "HTTP range response uses a non-identity content encoding",
        ))
    }
}

fn parse_response_content_range(headers: &HeaderMap) -> io::Result<ParsedContentRange> {
    let value = headers
        .get(CONTENT_RANGE)
        .ok_or_else(|| invalid_response("HTTP range response is missing Content-Range"))?
        .to_str()
        .map_err(|_| invalid_response("HTTP range response has an invalid Content-Range"))?;
    parse_content_range(value)
}

fn parse_content_range(value: &str) -> io::Result<ParsedContentRange> {
    let value = value
        .strip_prefix("bytes ")
        .ok_or_else(|| invalid_response("Content-Range unit is not bytes"))?;
    let (range, total) = value
        .split_once('/')
        .ok_or_else(|| invalid_response("Content-Range is malformed"))?;
    let (start, inclusive_end) = range
        .split_once('-')
        .ok_or_else(|| invalid_response("Content-Range byte range is malformed"))?;
    let start = start
        .parse::<u64>()
        .map_err(|_| invalid_response("Content-Range start is invalid"))?;
    let inclusive_end = inclusive_end
        .parse::<u64>()
        .map_err(|_| invalid_response("Content-Range end is invalid"))?;
    let total = total
        .parse::<u64>()
        .map_err(|_| invalid_response("Content-Range total length is invalid"))?;
    let end = inclusive_end
        .checked_add(1)
        .ok_or_else(|| invalid_response("Content-Range end overflows"))?;
    if start >= end || end > total {
        return Err(invalid_response("Content-Range bounds are inconsistent"));
    }
    Ok(ParsedContentRange { start, end, total })
}

fn response_validator(headers: &HeaderMap) -> Option<Validator> {
    if let Some(value) = headers.get(ETAG)
        && !value.as_bytes().starts_with(b"W/")
    {
        return Some(Validator {
            name: ETAG,
            value: value.clone(),
        });
    }
    headers.get(LAST_MODIFIED).map(|value| Validator {
        name: LAST_MODIFIED,
        value: value.clone(),
    })
}

fn read_exact_response(
    response: &mut ureq::http::Response<ureq::Body>,
    expected: usize,
) -> io::Result<Vec<u8>> {
    let limit = u64::try_from(expected)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "HTTP response is too large"))?
        .checked_add(1)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "HTTP response is too large"))?;
    let mut bytes = Vec::with_capacity(expected);
    response
        .body_mut()
        .as_reader()
        .take(limit)
        .read_to_end(&mut bytes)?;
    if bytes.len() != expected {
        return Err(invalid_response(
            "HTTP range response body length does not match Content-Range",
        ));
    }
    Ok(bytes)
}

fn reject_managed_header(name: &HeaderName) -> io::Result<()> {
    if name == RANGE || name == IF_RANGE || name == ACCEPT_ENCODING {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Range, If-Range, and Accept-Encoding are managed by decv-network",
        ));
    }
    Ok(())
}

fn invalid_response(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

struct RedactedEndpoint;

impl fmt::Debug for RedactedEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("<redacted>")
    }
}

struct RedactedHeaders<'a>(&'a HeaderMap);

impl fmt::Debug for RedactedHeaders<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<_> = self
            .0
            .keys()
            .map(|name| {
                if name == AUTHORIZATION || name == PROXY_AUTHORIZATION || name == COOKIE {
                    "<sensitive>"
                } else {
                    name.as_str()
                }
            })
            .collect();
        formatter.debug_list().entries(names).finish()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{self, BufRead, BufReader, Write},
        net::{TcpListener, TcpStream},
        sync::{Arc, Mutex},
        thread,
    };

    use decv_core::MediaInput;

    use super::{HttpRangeInput, ParsedContentRange, parse_content_range};
    use crate::RangeCacheConfig;

    #[test]
    fn parses_content_range() {
        assert_eq!(
            parse_content_range("bytes 20-29/100").unwrap(),
            ParsedContentRange {
                start: 20,
                end: 30,
                total: 100,
            }
        );
        assert!(parse_content_range("items 20-29/100").is_err());
        assert!(parse_content_range("bytes 29-20/100").is_err());
        assert!(parse_content_range("bytes 20-100/100").is_err());
        assert!(parse_content_range("bytes */100").is_err());
    }

    #[test]
    fn rejects_headers_owned_by_the_range_transport() {
        let error = HttpRangeInput::builder("http://example.invalid/video.mp4")
            .header("Range", "bytes=10-20")
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn probes_and_reads_strict_ranges_with_auth_and_if_range() {
        let fixture: Arc<[u8]> = Arc::from(*b"abcdefghijkl");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let (url, server) = range_server(fixture.clone(), requests.clone(), 2);

        let input = HttpRangeInput::builder(url)
            .header("Authorization", "Bearer test-secret")
            .unwrap()
            .cache_config(RangeCacheConfig::new(
                4.try_into().unwrap(),
                2.try_into().unwrap(),
            ))
            .build()
            .unwrap();
        let mut output = [0; 3];
        assert_eq!(input.read_at(1, &mut output).unwrap(), 3);
        assert_eq!(&output, b"bcd");
        server.join().unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[0].contains("range: bytes=0-0"));
        assert!(requests[1].contains("range: bytes=0-3"));
        assert!(requests[0].contains("authorization: bearer test-secret"));
        assert!(requests[1].contains("if-range: \"fixture-v1\""));
        assert!(requests[1].contains("accept-encoding: identity"));
    }

    fn range_server(
        bytes: Arc<[u8]>,
        requests: Arc<Mutex<Vec<String>>>,
        request_count: usize,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for stream in listener.incoming().take(request_count) {
                serve_range(stream.unwrap(), &bytes, &requests).unwrap();
            }
        });
        (format!("http://{address}/video.mp4"), server)
    }

    fn serve_range(
        mut stream: TcpStream,
        bytes: &[u8],
        requests: &Mutex<Vec<String>>,
    ) -> io::Result<()> {
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut request = String::new();
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 || line == "\r\n" {
                break;
            }
            request.push_str(&line);
        }
        let request = request.to_ascii_lowercase();
        let range = request
            .lines()
            .find_map(|line| line.strip_prefix("range: bytes="))
            .and_then(|value| value.split_once('-'))
            .and_then(|(start, end)| Some((start.parse().ok()?, end.parse().ok()?)))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing test range"))?;
        requests.lock().unwrap().push(request);

        let (start, inclusive_end): (usize, usize) = range;
        let body = &bytes[start..=inclusive_end];
        write!(
            stream,
            "HTTP/1.1 206 Partial Content\r\n\
             Content-Range: bytes {start}-{inclusive_end}/{}\r\n\
             Content-Length: {}\r\n\
             ETag: \"fixture-v1\"\r\n\
             Connection: close\r\n\
             \r\n",
            bytes.len(),
            body.len()
        )?;
        stream.write_all(body)?;
        stream.flush()
    }
}
