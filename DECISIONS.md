# Decisions

## Forecast one hour ahead, in 15-minute blocks

Demand charges are set by the highest 15-minute block in the month. What an operator can do about a coming peak is short-term: hold a furnace charge, stagger two motors, delay a compressor. That needs a warning minutes to an hour ahead, not a day-ahead forecast. So the targets are the next four blocks (15, 30, 45 and 60 minutes), forecast at the end of every block.

## Chronological split: train Jan-Sep, calibrate Oct, test Nov-Dec

A random split lets the model see the block before and after the one it predicts, which is interpolation, not forecasting. The test months come after everything the model has seen. October is held back for two jobs that must not use test data: early stopping for the neural networks, and the conformal interval widths.

The cost: Nov-Dec follow the winter tariff pattern, and October does not. The 90% intervals still cover 91-92% in the test. The 80% intervals cover only 70-78%, and the README reports that rather than widening them after looking at the test.

## Same-block measurements are excluded, and the leak is measured

Reactive energy, power factor and CO2 are measured in the same 15 minutes as the demand being predicted. CO2 is demand × 0.00045 rounded to two decimals, and the lagging power factor is computed exactly from kWh and kVArh. With them and a random split, gradient boosting reaches R² 0.998. That is the figure to beat in most write-ups of this dataset, and it is not a forecast. The test `test_features_never_see_the_block_being_forecast` changes demand after the forecast origin and checks that no feature moves.

## Load_Type is allowed

It looks like it could be derived from the load. It is not. Every day in the file follows exactly one of three patterns (spring-autumn, winter, and all-light for Sundays and holidays). That makes it a time-of-use calendar, known in advance, and it doubles as a holiday calendar. A test checks the three-pattern property.

## The midnight block belongs at the end of the day

The raw file stamps each day's last block with that day's date at 00:00. Read literally, 365 blocks sit at the start of their day, 24 hours early. Two independent signs say otherwise:
- the file order: the row comes after 23:45
- the readings: they differ from their file neighbours by 1.3 kWh on average, and from the same date's 00:15 by 4.2 kWh

A one-day shift fixes the timeline, which then runs on a perfect 15-minute grid from 00:15 on 1 Jan 2018 to 00:00 on 1 Jan 2019.

## No MAPE

Night load is 10-20 kW. A 10 kW miss there is a 50-100% error, while the same miss at a 400 kW peak is 2.5%. MAPE would rank models by how they do at 3 a.m. WAPE, MAE, and MAE and bias on the blocks above the limit are reported instead.

## The demand limit is the 95th percentile of training demand (403 kW)

The dataset has no contract demand. The limit stands in for the level a plant would want to stay under, and was fixed from training data before any test scoring. The 99th percentile (493 kW) is reported too, so the conclusions don't hang on one choice.

## Alerting on the upper bound, not the point forecast

Every model under-predicts the peaks: the mean error on blocks above the limit is −67 kW at 15 minutes and −98 kW at one hour. A peak is the unusual event a squared-error model is trained to hedge away from. Warning when the point forecast crosses the limit therefore misses most crossings, so the alert uses the conformal upper bound. The interval level is the operating knob: higher catches more and warns falsely more often. The README shows the trade-off at 80, 90 and 95%.

## Conformal widths per tariff band

Errors at night are a few kW, and errors in the maximum-load band are tens. One width for all blocks would be far too wide at night and too narrow at peak. Split-conformal applied separately per band (a Mondrian split) keeps the coverage guarantee within each band.

## "Shaved peak" is an upper bound on the benefit, and is labelled so

The monthly-peak figure assumes every warned-about crossing could be held at the limit by deferring load. Real plants can defer some loads and not others. The figure answers "what does the alert make possible", not "what would have happened". No rupee value is given: the dataset is from a Korean plant, and any tariff would be an assumption.
