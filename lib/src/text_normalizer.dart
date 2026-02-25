/// Stateless text normalization utilities used before WordPiece tokenization.
///
/// The normalization pipeline mirrors the pre-processing applied during model
/// training, so the same surface form is produced at both training and
/// inference time. Changing the pipeline without retraining the model will
/// degrade embedding quality.
///
/// Pipeline (when [normalize] is called):
/// 1. **Trim** leading / trailing whitespace.
/// 2. **Lowercase** all characters.
/// 3. **Remove punctuation** — non-word characters are replaced with a space.
/// 4. **Strip stopwords** — common function words are removed.
/// 5. **Collapse spaces** — multiple consecutive spaces become one.
///
/// If all words are stopwords (result would be empty), the original lowercased
/// text is returned as a fallback to preserve at least some signal.
///
/// Example:
/// ```dart
/// const normalizer = TextNormalizer(stopwords: {'what', 'is', 'the'});
/// normalizer.normalize('What IS the Dart language?');
/// // → 'dart language'
/// ```
class TextNormalizer {
  /// Creates a [TextNormalizer] with the given [stopWords].
  ///
  /// [stopWords] should contain lowercase tokens. Matching is case-insensitive
  /// because input is lowercased before comparison.
  const TextNormalizer({this.stopWords = const <String>{}});

  /// Set of lowercase words removed from input before tokenization.
  ///
  /// Matching is performed after the input has been lowercased, so entries
  /// must be in lowercase (e.g. `'the'`, not `'The'`).
  final Set<String> stopWords;

  // ---------------------------------------------------------------------------
  // Regex patterns
  // ---------------------------------------------------------------------------

  /// Matches any character that is not a Unicode letter, digit, or underscore.
  ///
  /// Uses the `\p{L}` (letter) and `\p{N}` (number) Unicode categories so
  /// that non-ASCII text (e.g. Cyrillic, CJK) is handled correctly.
  static final RegExp _nonWordChars = RegExp(
    r'[^\p{L}\p{N}_\s]',
    unicode: true,
  );

  /// Matches one or more whitespace characters.
  static final RegExp _whitespace = RegExp(r'\s+');

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /// Normalizes [text] by lowercasing, removing punctuation, and stripping
  /// stopwords.
  ///
  /// Returns the normalized string. If the result is empty (all words were
  /// stopwords), falls back to the trimmed and lowercased original text.
  ///
  /// This method is pure and stateless — it has no side effects and always
  /// returns the same output for the same input.
  String normalize(String text) {
    if (text.isEmpty) return text;

    // Step 1 & 2: trim + lowercase.
    final String lower = text.trim().toLowerCase();

    // Step 3: replace punctuation / symbols with spaces.
    final String cleaned = lower.replaceAll(_nonWordChars, ' ');

    // Step 4 & 5: split on whitespace, remove stopwords, rejoin.
    final List<String> words = cleaned
        .split(_whitespace)
        .where((String w) => w.isNotEmpty && !stopWords.contains(w))
        .toList();

    final String result = words.join(' ');

    // Fallback: if all words were stopwords, return the lowercased original.
    return result.isEmpty ? lower : result;
  }

  /// Splits [text] into individual content words after normalization.
  ///
  /// Equivalent to calling [normalize] and splitting on whitespace.
  /// Useful for keyword-overlap scoring or FTS query building.
  ///
  /// Example:
  /// ```dart
  /// const normalizer = TextNormalizer(stopwords: {'how', 'does'});
  /// normalizer.contentWords('How does Dart handle async?');
  /// // → ['dart', 'handle', 'async']
  /// ```
  List<String> contentWords(String text) {
    final String normalized = normalize(text);
    return normalized
        .split(_whitespace)
        .where((String w) => w.isNotEmpty)
        .toList();
  }

  /// Returns `true` if [word] is in the stopword list.
  ///
  /// The comparison is case-sensitive. Words should be lowercased before
  /// calling this method (the [normalize] pipeline does so automatically).
  bool isStopWord(String word) => stopWords.contains(word);

  @override
  String toString() =>
      'TextNormalizer(stopwords: ${stopWords.length} words)';
}
