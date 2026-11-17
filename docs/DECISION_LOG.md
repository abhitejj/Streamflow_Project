# Project decisions

These are the main choices used in the experiment.

| Decision | Why |
| --- | --- |
| Study 12 gauges across three states. | Compare different river conditions. Some gauges share rivers or are nested, so they are not independent examples. |
| Predict 1, 2, 4 and 8 weeks ahead. | Test how forecast usefulness changes with lead time. |
| Use a chronological split. | Train on 1980–2010, choose settings on 2011–2015, and evaluate on 2016–2025. Remove the last H training and validation rows for each H-week lead. |
| Keep missing observations missing. | Do not create target discharge values using later measurements. A weekly target needs at least four original daily measurements. |
| Retain gauges with less than 15% missing daily discharge. | Keep all 12 gauges without filling gaps. Report actual usable weeks and drought-event counts separately. |
| Exclude configured unmeasured zeros. | At Karad, Warunji, Yadgir and Huvinhedigi, recorded-zero issue or target flow is excluded under the documented monitoring policy. This interpretation has limits. |
| Fit preprocessing on training data only. | Fit weather corrections on 2001–2010 and save training medians, scaling and feature order with each model. |
| Compare simple and trained models. | Use persistence at 1–2 weeks and weekly climatology at 4–8 weeks. Compare flow-only models with models that also use weather. |
| Evaluate drought warnings separately. | Accurate discharge forecasts do not necessarily detect low-flow events. Freeze drought thresholds from training and report hits, misses and false alarms. |
| Use the supplied sources with their actual labels. | Later temperature and reference ET0 use Open-Meteo default Best Match, not verified pure ERA5-Land. Karnataka discharge was recovered from the merged daily records. |
| Publish a static website. | Let readers inspect saved results without running training in their browser. |

## Implementation and limitations

The experiment uses explicit preceding-week windows, no daily interpolation, training-only source harmonization and saved model preprocessing. Dataset and model hashes link the reported results to their inputs.

The weather-product mixture, missing observations, few drought events, related gauges and unverified real-time data availability remain limitations. Comparisons between different pipeline versions cannot isolate the effect of one methodological change.

See the [report](PROJECT_REPORT_DOCUMENTATION.md) for methods and evidence, and the [README reproduction instructions](https://github.com/abhitejj/Streamflow_Project#reproduction-and-validation) for the command sequence.
