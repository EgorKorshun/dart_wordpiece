/// Defines the set of special tokens used by a WordPiece tokenizer.
///
/// Special tokens serve structural roles in BERT-style models:
/// - [cls] marks the beginning of every sequence.
/// - [sep] separates segments and marks the end of each sequence.
/// - [pad] fills positions beyond the real content up to [TokenizerConfig.maxLength].
/// - [unk] replaces any sub-string that cannot be found in the vocabulary.
/// - [subwordPrefix] is prepended to continuation pieces (e.g. `##ing`).
///
/// Use [SpecialTokens.bert] for standard BERT/DistilBERT/TinyBERT vocabularies.
/// Supply a custom instance when working with non-standard tokenizer setups.
///
/// Example:
/// ```dart
/// // Standard BERT tokens (default).
/// const tokens = SpecialTokens.bert();
///
/// // Custom tokens for a domain-specific model.
/// const tokens = SpecialTokens(
///   cls: '<s>',
///   sep: '</s>',
///   pad: '<pad>',
///   unk: '<unk>',
///   subwordPrefix: '##',
/// );
/// ```
class SpecialTokens {
  /// Creates a custom set of special tokens.
  const SpecialTokens({
    required this.cls,
    required this.sep,
    required this.pad,
    required this.unk,
    required this.subwordPrefix,
  });

  /// Standard BERT special tokens as defined in the original paper and
  /// `bert-base-uncased` / `bert-base-cased` vocabularies.
  ///
  /// - `[CLS]` — classification / start-of-sequence token.
  /// - `[SEP]` — separator token placed after each segment.
  /// - `[PAD]` — padding token; attention mask is `0` for pad positions.
  /// - `[UNK]` — replaces out-of-vocabulary sub-strings.
  /// - `##` — subword continuation prefix (e.g. `play` + `##ing`).
  const SpecialTokens.bert()
      : cls = '[CLS]',
        sep = '[SEP]',
        pad = '[PAD]',
        unk = '[UNK]',
        subwordPrefix = '##';

  /// Start-of-sequence / classification token prepended to every input.
  ///
  /// In BERT, the hidden state of [cls] is used as the aggregate sequence
  /// representation for classification tasks.
  final String cls;

  /// Separator token appended after each segment.
  ///
  /// Single-sequence input:  `[CLS] tokens [SEP]`
  /// Sentence-pair input:    `[CLS] A [SEP] B [SEP]`
  final String sep;

  /// Padding token used to fill sequences shorter than [TokenizerConfig.maxLength].
  ///
  /// The attention mask is set to `0` for every pad position so that the model
  /// ignores them during self-attention.
  final String pad;

  /// Unknown token substituted for any sub-string absent from the vocabulary.
  ///
  /// If the greedy longest-match WordPiece search finds no match for a
  /// character span, the entire remaining word is replaced with [unk].
  final String unk;

  /// Prefix added to every non-initial subword piece.
  ///
  /// BERT uses `##` (e.g. the word `"playing"` → `["play", "##ing"]`).
  /// Some models use `Ġ` (space prefix, RoBERTa style) or other markers.
  final String subwordPrefix;

  @override
  String toString() =>
      'SpecialTokens(cls: $cls, sep: $sep, pad: $pad, unk: $unk, prefix: $subwordPrefix)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is SpecialTokens &&
          cls == other.cls &&
          sep == other.sep &&
          pad == other.pad &&
          unk == other.unk &&
          subwordPrefix == other.subwordPrefix;

  @override
  int get hashCode =>
      Object.hash(cls, sep, pad, unk, subwordPrefix);
}
