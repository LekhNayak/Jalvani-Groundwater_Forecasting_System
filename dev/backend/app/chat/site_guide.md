# JalVaani site guide (given to the assistant; keep in sync with dev/frontend)

JalVaani shows groundwater levels and forecasts for the districts of Maharashtra, India. It has two views:
- the farmer view, for one district;
- the government view, for the whole state.

The data comes from:
- CGWB monitoring wells (groundwater);
- CHIRPS (rainfall);
- ERA5-Land (soil moisture);
- the CGWB groundwater resource assessment (extraction).

The forecasts come from the project's model, a Temporal Fusion Transformer ensemble. It was tested on two years of past data.

## Pages (routes)
- `/`: start page. Choose a language and a role (farmer or government officer).
- `/farmer`: farmer view. Pick a district first.
- `/farmer/<district-slug>`: the farmer view for one district.
- `/gov`: government overview of all 36 districts.
- `/gov/district/<district-slug>`: the government view of one district.
- `/help`: how to use the site, a glossary, and data sources and limits.

District slugs are lower case, with spaces as underscores (for example `pune`, `mumbai_suburban`).

## Elements you can highlight (element ids)

**Every page:**
- `lang-switch`: language buttons
- `role-switch`: change role
- `help-link`: help page
- `freshness-bar`: when each kind of data was last updated
- `chat-button`: this assistant

**Farmer view:**

| id | What it shows |
|---|---|
| `district-picker` | choose a district |
| `card-water-now` | the latest groundwater level (metres below ground) and how it compares with the same season in past years |
| `card-forecast` | expected level after 7, 14 and 30 days, with a narrow "most likely" range and a wide "could be" range |
| `reliability-badge` | how well past forecasts did in this district (good / fair / limited) |
| `card-rain-soil` | rain since 1 June compared with the 2023–2025 average, and soil moisture |
| `advice-list` | general water-use tips that follow from the numbers |
| `details-chart` | the past levels and the forecast, as a chart |
| `listen-button` | read a card aloud (only when voice is available) |

**Government view:**

| id | What it shows |
|---|---|
| `kpi-row` | monitored districts, wells, extraction categories, data freshness |
| `map` | a district map, coloured by the chosen layer; tapping a district shows it in `glance-panel` |
| `glance-panel` | "District at a glance": the selected district's key numbers and reasons, with a button to open its page |
| `map-layers` | choose the map layer: level vs usual, 30-day forecast change, stage of extraction, monitoring wells, forecast reliability |
| `priority-table` | districts ranked by rule-based flags, each with its reason |
| `needs-panel` | what data and monitoring exist, and what is missing |
| `download-csv` | download the priority table |

**Government district page:**
- `gov-history-chart`: levels and past forecasts with ranges
- `gov-extraction`: extraction history and how the water is used
- `gov-wells`: monitoring wells
- `gov-metrics`: forecast accuracy

## Guided tours
- `farmer`: walks through the farmer view.
- `government`: walks through the government view.

## Glossary
- **Groundwater level, m below ground:** how deep the water is in the wells. A bigger number means deeper water, which is worse.
- **Falling / rising:** the water level is going deeper or coming up.
- **Compared with usual:** today's level against the same time of year in 2023–2025. The records start in 2023, so "usual" means only those years.
- **Forecast, "most likely" and "could be":** the expected level, with two ranges around it. The darker "most likely" range is the typical size of past errors in that district, so the true level falls inside it about half the time. The lighter "could be" range is wider: the true level falls inside it about 8 times out of 10. Both belong to the same forecast.
- **Week average vs the last day:** "Water in the wells now" shows the average of the last reading week. The forecast starts from the last single day's reading, which can differ by up to about 1 m in a few districts; both are labelled on the page.
- **Reliability:** how close past forecasts in this district were, compared with simply assuming "no change". It is "good", "fair" or "limited".
- **Stage of extraction:** groundwater used ÷ groundwater that can be used each year, from CGWB. Safe is below 70%, semi-critical 70–90%, critical 90–100%, over-exploited above 100%.
- **Provisional:** recent rainfall that will be revised when the final data is published.
- **No monitoring data:** CGWB publishes no usable well readings for this district, so there is no level or forecast. Rain and extraction figures are still shown.

## Limits (say these honestly when asked)
- **District level only.** There is no village or taluka detail.
- **Groundwater readings end on the date shown in `freshness-bar`.** Newer readings have not been published by CGWB yet. The levels shown are from that date, and the latest forecast may cover dates that have already passed. Say so when asked about today's level.
- **The tips are general guidance.** For crop, borewell, scheme or legal questions, the user should contact the local agriculture office, the Groundwater Surveys and Development Agency (GSDA), or the district administration.
