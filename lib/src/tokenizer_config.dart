import 'special_tokens.dart';

/// Configuration options for [WordPieceTokenizer].
///
/// All fields have sensible defaults that match the standard BERT-base
/// configuration, so you only need to specify what differs from the baseline.
///
/// Example — default BERT config:
/// ```dart
/// const config = TokenizerConfig(); // maxLength=64, BERT tokens, no stopwords
/// ```
///
/// Example — custom sequence length and stopword filtering:
/// ```dart
/// final config = TokenizerConfig(
///   maxLength: 128,
///   stopwords: {'is', 'the', 'a', 'an'},
///   normalizeText: true,
/// );
/// ```
class TokenizerConfig {
  /// Creates a tokenizer configuration.
  ///
  /// [maxLength] — total number of token slots including [SpecialTokens.cls]
  ///   and [SpecialTokens.sep]. Sequences longer than this are truncated;
  ///   shorter sequences are padded. Must be ≥ 2. Defaults to `64`.
  ///
  /// [specialTokens] — special token strings used during encoding.
  ///   Defaults to standard BERT tokens (`[CLS]`, `[SEP]`, etc.).
  ///
  /// [stopwords] — set of lowercase words removed from input before
  ///   tokenization when [normalizeText] is `true`. Defaults to an empty set
  ///   (no filtering). Useful when the model was trained with stopword removal.
  ///
  /// [normalizeText] — when `true`, input text is lowercased, non-word
  ///   characters are removed, and [stopwords] are stripped before the
  ///   WordPiece algorithm runs. Set to `false` to pass raw text unchanged
  ///   (required for cased models). Defaults to `true`.
  const TokenizerConfig({
    this.maxLength = 64,
    this.specialTokens = const SpecialTokens.bert(),
    this.stopwords = const <String>{},
    this.normalizeText = true,
  }) : assert(maxLength >= 2, 'maxLength must be at least 2 (CLS + SEP)');

  /// Maximum number of token IDs in the encoded output, including special
  /// tokens ([SpecialTokens.cls] and [SpecialTokens.sep]).
  ///
  /// Sequences that exceed [maxLength] after tokenization are truncated so
  /// that the final token is always [SpecialTokens.sep].
  /// Sequences shorter than [maxLength] are padded with [SpecialTokens.pad].
  final int maxLength;

  /// Set of special tokens used during encoding and decoding.
  ///
  /// Override this only when working with non-standard vocabularies.
  /// For BERT-base-uncased / BERT-base-cased use the default value.
  final SpecialTokens specialTokens;

  /// Lowercase words to remove from input text before tokenization.
  ///
  /// Stopword removal is applied only when [normalizeText] is `true`.
  /// The set is matched after lowercasing, so entries should be lowercase.
  ///
  /// Example — common English function words:
  /// ```dart
  /// stopwords: {'what', 'is', 'are', 'the', 'a', 'an', 'how', 'why'},
  /// ```
  final Set<String> stopwords;

  /// Whether to normalize input text before tokenization.
  ///
  /// When `true` (default):
  /// 1. Text is trimmed and lowercased.
  /// 2. Non-word characters (punctuation, symbols) are replaced with spaces.
  /// 3. Words in [stopwords] are removed.
  ///
  /// Set to `false` for cased models (e.g. `bert-base-cased`) or when your
  /// application performs its own pre-processing.
  final bool normalizeText;

  @override
  String toString() => 'TokenizerConfig('
      'maxLength: $maxLength, '
      'normalizeText: $normalizeText, '
      'stopwords: ${stopwords.length} words, '
      'specialTokens: $specialTokens'
      ')';
}
