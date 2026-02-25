## 1.0.0

Initial release.

### Added
- `WordPieceTokenizer` — BERT-compatible WordPiece tokenizer in pure Dart.
- `encode(text)` — single-sequence encoding with padding and truncation.
- `encodePair(textA, textB)` — sentence-pair encoding with `token_type_ids`.
- `encodeAll(texts)` — batch encoding.
- `tokenize(text)` — returns raw token strings without padding.
- `TokenizerOutput` — typed result with `inputIds`, `attentionMask`,
  `tokenTypeIds`, `Int64List` getters for ONNX tensor creation.
- `TokenizerConfig` — configurable `maxLength`, `stopwords`,
  `normalizeText`, and `SpecialTokens`.
- `SpecialTokens` — default `SpecialTokens.bert()` and custom constructor.
- `TextNormalizer` — standalone lowercase / punctuation / stopword normalizer.
- `VocabLoader` — vocabulary loading from `dart:io` File, String, or Map.
