## 1.2.0

### Added
- `TokenizerOutput.offsetMapping` — `List<(int, int)>?` of `(start, end)`
  character spans in the normalized text for each token position.
  Special tokens ([CLS], [SEP], [PAD]) receive sentinel `(0, 0)` spans.
  Enables NER, QA span extraction, and text highlighting in Flutter apps.
- `TokenizerOutput.specialTokensMask` — `List<int>?` where `1` marks
  [CLS]/[SEP]/[PAD] and `0` marks real content tokens.
- `TokenizerOutput.overflowingTokens` — `List<String>?` of tokens removed
  by truncation, in order. Useful for sliding-window strategies.
- `PaddingStrategy` enum — `fixed` (default, pads to `maxLength`) and
  `longest` (pads to longest sequence in batch). Applies to `encodeAll()`.
- `TruncationSide` enum — `right` (default) and `left`.
- `TokenizerConfig.paddingStrategy` and `TokenizerConfig.truncationSide`
  fields (both optional; defaults reproduce v1.1.0 behavior exactly).
- `encodeAsync()`, `encodePairAsync()`, `encodeAllAsync()` — async variants
  for Flutter `compute()`/isolate integration.

### Changed
- Internal `_wordpieceEncode` now returns `(tokens, offsets)` instead of
  mutating a caller-provided list.

### Backward compatibility
- All new `TokenizerOutput` fields are optional (`null` default) — existing
  call sites that pass only the three required constructor arguments compile
  and behave identically to v1.1.0.
- All new `TokenizerConfig` fields have defaults matching v1.1.0 behavior.
- No existing public method signatures were changed.

---

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
