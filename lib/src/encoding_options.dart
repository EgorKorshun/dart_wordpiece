/// Controls how sequences are padded during encoding.
enum PaddingStrategy {
  /// Pad every sequence to [TokenizerConfig.maxLength].
  ///
  /// This is the default and reproduces v1.1.0 behavior exactly.
  fixed,

  /// Pad every sequence in a batch to the length of the longest sequence
  /// in that batch.
  ///
  /// Applies only to [WordPieceTokenizer.encodeAll] (and its async variant).
  /// Single-sequence [encode] and [encodePair] always use [fixed] padding.
  /// Saves compute and memory when sequences vary greatly in length.
  longest,
}

/// Controls which side of a sequence is truncated when it exceeds
/// [TokenizerConfig.maxLength].
enum TruncationSide {
  /// Truncate tokens from the end (right side). Default, matches v1.1.0.
  right,

  /// Truncate tokens from the start (left side).
  ///
  /// Useful for models that focus on the end of long documents, e.g.
  /// last-N-tokens classification strategies.
  left,
}
