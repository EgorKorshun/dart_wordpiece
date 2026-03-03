## 1.1.0

### Changed
- `idToToken(id)` now uses an internal reverse-map built at construction time —
  lookup is O(1) instead of O(n).

### Added
- Example 8: `tokenToId` / `idToToken` / `vocabSize` usage.
- Example 9: `Int64List` tensor conversion for ONNX Runtime.

---

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
