import 'dart:typed_data';

/// The result of encoding one or two text sequences with [WordPieceTokenizer].
///
/// All three core lists share the same length — exactly
/// [TokenizerConfig.maxLength] (or the batch-longest length when using
/// [PaddingStrategy.longest]).
///
/// ```
/// Input:  "Flutter is fast"   (maxLength = 8)
///
/// Tokens: [CLS] flutter is fast [SEP] [PAD] [PAD] [PAD]
///
/// inputIds       : [101, 7688, 2003, 3435, 102,   0,   0,   0]
/// attentionMask  : [  1,    1,    1,    1,   1,   0,   0,   0]
/// tokenTypeIds   : [  0,    0,    0,    0,   0,   0,   0,   0]
/// offsetMapping  : [(0,0),(0,7),(8,10),(11,15),(0,0),(0,0),(0,0),(0,0)]
/// specialTokensMask:[  1,    0,    0,    0,   1,   1,   1,   1]
/// ```
///
/// For sentence-pair input (see [WordPieceTokenizer.encodePair]):
///
/// ```
/// Input A: "Flutter"   Input B: "fast"   (maxLength = 10)
///
/// Tokens: [CLS] flutter [SEP] fast [SEP] [PAD] [PAD] [PAD] [PAD] [PAD]
///
/// inputIds       : [101, 7688, 102, 3435, 102,  0,  0,  0,  0,  0]
/// attentionMask  : [  1,    1,   1,    1,   1,  0,  0,  0,  0,  0]
/// tokenTypeIds   : [  0,    0,   0,    1,   1,  0,  0,  0,  0,  0]
/// ```
class TokenizerOutput {
  /// Creates a tokenizer output.
  ///
  /// [inputIds], [attentionMask], and [tokenTypeIds] are required and must
  /// share the same length.  The remaining fields are optional and will be
  /// populated by [WordPieceTokenizer] automatically.
  const TokenizerOutput({
    required this.inputIds,
    required this.attentionMask,
    required this.tokenTypeIds,
    this.offsetMapping,
    this.specialTokensMask,
    this.overflowingTokens,
  });

  /// Vocabulary IDs for every token position.
  ///
  /// - Index 0 is always the [SpecialTokens.cls] token ID.
  /// - The last real token is always [SpecialTokens.sep].
  /// - Trailing positions contain the [SpecialTokens.pad] token ID.
  ///
  /// Pass this directly to an ONNX BERT model as the `input_ids` tensor.
  final List<int> inputIds;

  /// Binary mask indicating real (`1`) vs. padding (`0`) positions.
  ///
  /// Pass this to an ONNX BERT model as the `attention_mask` tensor.
  /// The model ignores positions where the mask is `0`.
  final List<int> attentionMask;

  /// Segment IDs distinguishing the first (`0`) and second (`1`) sequences.
  ///
  /// For single-sequence encoding all values are `0`.
  /// For sentence-pair encoding ([WordPieceTokenizer.encodePair]):
  /// - Tokens belonging to sequence A (including its `[CLS]` and `[SEP]`) → `0`.
  /// - Tokens belonging to sequence B (including its `[SEP]`) → `1`.
  /// - Padding positions → `0`.
  ///
  /// Pass this to an ONNX BERT model as the `token_type_ids` tensor.
  final List<int> tokenTypeIds;

  /// Character-span offsets in the **normalized** text for each token position,
  /// as `(start, end)` record pairs (end is exclusive).
  ///
  /// Special tokens ([CLS], [SEP], [PAD]) receive the sentinel span `(0, 0)`.
  /// Subword continuation pieces (e.g. `##ing`) receive the span covering
  /// their characters within the original word, without the `##` prefix.
  ///
  /// Example — `"playing"` (7 chars) at the start of the sequence:
  /// ```
  /// play  → (0, 4)
  /// ##ing → (4, 7)
  /// ```
  ///
  /// Offsets are relative to the normalized string produced by
  /// [TextNormalizer], not the raw input. This is the same convention used
  /// by HuggingFace Tokenizers in Python.
  ///
  /// Length equals [length] (same as [inputIds]).
  final List<(int, int)>? offsetMapping;

  /// Binary mask where `1` marks special tokens ([CLS], [SEP], [PAD]) and
  /// `0` marks real content tokens.
  ///
  /// Useful for masked language model post-processing and attention
  /// manipulation. Length equals [length].
  final List<int>? specialTokensMask;

  /// Tokens that were removed by truncation, in their original order.
  ///
  /// Empty list when no truncation occurred.
  /// Useful for implementing sliding-window strategies over long documents.
  final List<String>? overflowingTokens;

  /// Total number of token slots (equals [TokenizerConfig.maxLength], or the
  /// batch-longest length with [PaddingStrategy.longest]).
  int get length => inputIds.length;

  /// Number of real (non-padding) token positions.
  ///
  /// Equals the number of positions where [attentionMask] is `1`.
  /// Use this to determine the actual sequence length before padding.
  int get realLength => attentionMask.fold(0, (sum, v) => sum + v);

  /// Returns [inputIds] as an [Int64List] ready for ONNX tensor creation.
  ///
  /// Example with `package:onnxruntime`:
  /// ```dart
  /// final tensor = OrtValueTensor.createTensorWithDataList(
  ///   output.inputIdsInt64,
  ///   [1, output.length],
  /// );
  /// ```
  Int64List get inputIdsInt64 => Int64List.fromList(inputIds);

  /// Returns [attentionMask] as an [Int64List] ready for ONNX tensor creation.
  Int64List get attentionMaskInt64 => Int64List.fromList(attentionMask);

  /// Returns [tokenTypeIds] as an [Int64List] ready for ONNX tensor creation.
  Int64List get tokenTypeIdsInt64 => Int64List.fromList(tokenTypeIds);

  @override
  String toString() {
    final buffer = StringBuffer(
      'TokenizerOutput('
      'length: $length, '
      'realLength: $realLength, '
      'inputIds: $inputIds',
    );
    if (offsetMapping != null) buffer.write(', offsetMapping: $offsetMapping');
    if (specialTokensMask != null) {
      buffer.write(', specialTokensMask: $specialTokensMask');
    }
    if (overflowingTokens != null && overflowingTokens!.isNotEmpty) {
      buffer.write(', overflowingTokens: $overflowingTokens');
    }
    buffer.write(')');
    return buffer.toString();
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is TokenizerOutput &&
          _listEquals(inputIds, other.inputIds) &&
          _listEquals(attentionMask, other.attentionMask) &&
          _listEquals(tokenTypeIds, other.tokenTypeIds) &&
          _offsetEquals(offsetMapping, other.offsetMapping) &&
          _listEquals(
            specialTokensMask ?? const [],
            other.specialTokensMask ?? const [],
          ) &&
          _stringListEquals(
            overflowingTokens ?? const [],
            other.overflowingTokens ?? const [],
          );

  @override
  int get hashCode => Object.hash(
        Object.hashAll(inputIds),
        Object.hashAll(attentionMask),
        Object.hashAll(tokenTypeIds),
        offsetMapping == null ? null : Object.hashAll(offsetMapping!),
        specialTokensMask == null
            ? null
            : Object.hashAll(specialTokensMask!),
        overflowingTokens == null
            ? null
            : Object.hashAll(overflowingTokens!),
      );

  static bool _listEquals(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  static bool _stringListEquals(List<String> a, List<String> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  static bool _offsetEquals(List<(int, int)>? a, List<(int, int)>? b) {
    if (a == null && b == null) return true;
    if (a == null || b == null) return false;
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i].$1 != b[i].$1 || a[i].$2 != b[i].$2) return false;
    }
    return true;
  }
}
