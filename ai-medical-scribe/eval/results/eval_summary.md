## Note quality on ACI-Bench `valid` (20 cases)

| Model | Valid notes | Lines/note | Invented facts | Missing facts | Verifier red / amber / green | Gen. time |
|---|---|---|---|---|---|---|
| qwen2.5:0.5b | 20/20 | 11.8 | 65/236 (28%) | 546/614 (89%) | 85% / 11% / 5% | 103 s |
| qwen2.5:1.5b | 20/20 | 6.7 | 37/134 (28%) | 538/614 (88%) | 71% / 8% / 21% | 185 s |

**ROUGE-L (F1)** per section

| Model | S | O | A | P | Whole note |
|---|---|---|---|---|---|
| qwen2.5:0.5b | 0.150 | 0.120 | 0.170 | 0.148 | 0.249 |
| qwen2.5:1.5b | 0.166 | 0.036 | 0.213 | 0.178 | 0.232 |

**BERTScore F1 (rescaled, `distilbert-base-uncased`)** per section

| Model | S | O | A | P | Whole note |
|---|---|---|---|---|---|
| qwen2.5:0.5b | 0.252 | 0.161 | 0.278 | 0.239 | 0.319 |
| qwen2.5:1.5b | 0.325 | 0.048 | 0.331 | 0.279 | 0.341 |

**Missing facts by section** (share of reference sentences not covered)

| Model | S | O | A | P |
|---|---|---|---|---|
| qwen2.5:0.5b | 87% | 87% | 92% | 92% |
| qwen2.5:1.5b | 85% | 93% | 87% | 86% |

Speech-to-text (faster-whisper base.en, jiwer): pooled WER **4.8%** on 5 synthesised consultations (`eval/results/stt_wer.csv`).

Invented = note line not entailed by any transcript passage; missing = reference sentence not entailed by the generated note (local NLI, threshold 0.5). Automatic estimates, not clinician review.
