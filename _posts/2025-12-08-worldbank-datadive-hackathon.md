---
title: "Analysing the impact of AI jobs at World Bank DataDive 2025"
date: 2025-12-08
categories:
  - ai
  - data-science
  - labor
tags:
  - hackathon
  - machine-learning
  - dashboard
  - world-bank
  - gradio
  - economics
excerpt: "Our team built an interactive dashboard analyzing global AI job market trends and skills gaps across 66 countries using Stanford HAI data and World Bank labor indicators."
---

# Building JobsLens AI at World Bank DataDive 2025

Last week, I participated in the **World Bank DataDive 2025**, a 8-hour hackathon focused on extracting insights from global labor market data. Our team of five—Maryam Shahbaz Ali, Paul Suhwan Lee, Stevens Cadet, Yingquan Li, and myself—tackled [**Challenge 8: Digital/AI Job Demand & Supply Analysis**](https://www.dc2.org/datadive).

**TL;DR**: We built [JobsLens AI](https://huggingface.co/spaces/rkovashikawa/jobslens-ai), an interactive Gradio dashboard that visualizes AI job market trends across countries, identifies skills gaps, and forecasts future demand.

---

## The Challenge

The World Bank wanted to understand **where digital and AI job opportunities are rising or lagging** across countries, industries, and skill types. The goal was to help policymakers identify:

- Countries with high AI job demand but low skilled workforce supply (skills gaps)
- Emerging markets with growing AI sectors
- Trends in AI talent concentration and investment

---

## The Data

We worked with two primary datasets:

1. **Stanford HAI AI Index 2025** (528 observations, 232 columns)
   - Panel data spanning 2017-2024 for 66 countries
   - Key metrics: AI job postings, skills penetration, talent concentration, patents, investment
   - Target variable: `ai_job_postings_perc_of_all_job_postings`

2. **World Bank Global Labor Database** (154 countries, cross-sectional)
   - Employment rates by sector and education level
   - Labor force participation, unemployment rates
   - Demographics and population statistics

---

## Technical Approach

### Data Integration Challenge

The fundamental problem was joining **cross-sectional predictors** (World Bank) with **panel outcomes** (HAI):

- HAI: Multiple years per country (2017-2024)
- World Bank: Single observation per country

We used a **many-to-one merge**, assuming labor market characteristics remain relatively stable over time.

### Dashboard Development

Built with **Gradio 4.0** and **Plotly**, the dashboard features 5 interactive tabs:

1. **🌍 Global Overview**: Choropleth map showing supply-demand gaps
2. **📊 Country Comparison**: Multi-country trend analysis (default: Brazil, Argentina, USA, Japan)
3. **🏆 Rankings**: Top countries by AI job demand, skills supply, investment, startups
4. **🔮 Forecasts**: 2024-2025 predictions using Ridge regression
5. **🔬 Country Deep Dive**: Detailed statistics for individual countries

---

## Deployment

We deployed to **Hugging Face Spaces** for free hosting:

```bash
# Project structure
Team_Projects/JobsLens_AI/
├── app.py                    # Gradio dashboard
├── data/
│   ├── hai_full_database.csv
│   └── forecasts_2024_2025.csv
├── hf_space/                 # Git repo for HF Spaces
└── notebooks/
    └── EDA.ipynb
```

**Tech stack**:
- **Frontend**: Gradio 4.0 with custom Plotly visualizations
- **Data**: pandas, numpy
- **ML**: scikit-learn (Ridge, Random Forest, Gradient Boosting)
- **Deployment**: Hugging Face Spaces (free CPU tier)

---

## Key Insights from the Data

### Skills Gap Leaders (High Demand/Low Supply)
- **Switzerland**: 1.41% AI job demand (highest forecasted)
- **Netherlands**: 1.25%
- **United States**: 1.36%

### Supply-Demand Balance
The global AI job market shows:
- Growing demand outpacing skills supply in developed economies
- Emerging markets (Brazil, Mexico) showing moderate growth
- Investment concentrated in US, China, EU

### Temporal Trends (2017-2024)
- AI job postings grew ~3-5% annually in top markets
- Skills penetration lagging by 2-3 years
- COVID-19 accelerated remote AI hiring (2020-2021 spike)

---

## What I Learned

### Technical
1. **Small data ≠ deep learning**: With only 83 observations, simpler models (Ridge) outperformed complex ones
2. **Feature engineering matters**: Missing 14/24 features severely limited model performance
3. **Cross-sectional + panel merging**: Requires careful assumptions about temporal stability
4. **Gradio is powerful**: Built a full interactive dashboard in <2 hours

### Hackathon Strategy
1. **Iterate fast**: We went from EDA → model → dashboard → deployment in <8 hours
2. **Visualize early**: Interactive plots helped us spot data issues quickly
3. **Deploy early**: Having a live URL motivated us to polish the final product
4. **Team roles**: Divided into data/modeling (me + Yingquan), visualization (Paul), storytelling (Maryam), infrastructure (Stevens)

---

## Results & Recognition

- **Live Dashboard**: [jobslens-ai on HF Spaces](https://huggingface.co/spaces/rkovashikawa/jobslens-ai)
- **Presentation**: [Canva slides](https://www.canva.com/design/DAG6ftQIfSA/pW-b5By0racKOTOgHcUY6A/view)
- **Code**: [GitHub repository](https://github.com/datacommunitydc/DataDive25)

While our forecasting model didn't achieve production-ready accuracy, the **exploratory dashboard successfully visualizes historical trends** and provides policymakers with actionable insights on global AI skills gaps.

---

## Future Improvements

If I were to continue this project:

1. **Use "All" subsample** instead of "Urban" to get 24/24 features
2. **Time-series models**: ARIMA or exponential smoothing might outperform ML on small data
3. **External data**: Add OECD education stats, GitHub commit data by country
4. **Sector analysis**: Break down AI jobs by industry (healthcare, finance, etc.)
5. **Interactive filtering**: Let users select custom country groups and metrics

---

## Try It Yourself

🔗 **Live Dashboard**: [https://huggingface.co/spaces/rkovashikawa/jobslens-ai](https://huggingface.co/spaces/rkovashikawa/jobslens-ai)

Explore AI job trends for 66 countries, compare Brazil vs. USA, or dive into Switzerland's AI ecosystem. The dashboard is free and requires no login.

---

*Hackathon: World Bank DataDive 2025*
*Team: JobsLens AI*
*Duration: 8 hours*
*Tech: Python, Gradio, Plotly, scikit-learn, Hugging Face Spaces*
