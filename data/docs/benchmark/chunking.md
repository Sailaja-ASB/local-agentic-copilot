# Chunking Documents

Documents are split into chunks before embedding because models have input limits and because a small, focused passage is easier to match than a whole file. A typical fixed-size splitter uses around 500 characters per chunk with about 100 characters of overlap, so a sentence cut at a boundary still appears whole in one of the neighbouring chunks.

Chunks that are too large mix several topics and dilute the embedding; chunks that are too small lose context. Each chunk should keep metadata such as source file, page number and chunk index so answers can cite where the evidence came from.
