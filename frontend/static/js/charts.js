window.CycloneCharts = (() => {
  let scoreChart = null;

  const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    devicePixelRatio: 2,
  };

  function textColor() {
    return "#8d9baa";
  }

  function gridColor() {
    return "rgba(138, 159, 177, 0.08)";
  }

  function formatAxisTimestamp(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  }

  function baseScaleOptions() {
    return {
      x: {
        ticks: {
          color: textColor(),
          maxTicksLimit: 8,
          maxRotation: 0,
          callback(value, index, ticks) {
            const label = this.getLabelForValue(value);
            return formatAxisTimestamp(label);
          },
        },
        grid: { color: gridColor(), drawBorder: false },
        border: { display: false },
      },
      y: {
        ticks: { color: textColor(), maxTicksLimit: 7 },
        grid: { color: gridColor(), drawBorder: false },
        border: { display: false },
      },
    };
  }

  function tooltipTitle(context) {
    if (!context || !context[0]) return "";
    return formatAxisTimestamp(context[0].label);
  }

  function renderScore(records, threshold) {
    const canvas = document.getElementById("scoreChart");
    if (!canvas) return;

    if (scoreChart) scoreChart.destroy();

    const labels = records.map((record) => record.time_dt);
    const scores = records.map((record) => Number(record.anomaly_score));
    const thresholdValue = Number(threshold);

    scoreChart = new Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Anomaly score",
            data: scores,
            borderColor: "#44c7d9",
            backgroundColor: "rgba(68, 199, 217, 0.08)",
            borderWidth: 1.5,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: "#44c7d9",
            tension: 0.18,
            fill: true,
          },
          {
            label: "Anomaly threshold",
            data: labels.map(() => thresholdValue),
            borderColor: "#f5a623",
            backgroundColor: "transparent",
            borderWidth: 1.25,
            borderDash: [7, 5],
            pointRadius: 0,
            tension: 0,
          },
        ],
      },
      options: {
        ...chartDefaults,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            display: true,
            align: "end",
            labels: {
              color: textColor(),
              boxWidth: 18,
              boxHeight: 2,
              usePointStyle: false,
              padding: 14,
              font: { size: 10 },
            },
          },
          tooltip: {
            backgroundColor: "#151d26",
            borderColor: "#2d3a47",
            borderWidth: 1,
            titleColor: "#eef4f8",
            bodyColor: "#bac5ce",
            padding: 10,
            displayColors: true,
            callbacks: { title: tooltipTitle },
          },
        },
        scales: baseScaleOptions(),
      },
    });
  }

  function destroyAll() {
    if (scoreChart) {
      scoreChart.destroy();
      scoreChart = null;
    }
  }

  return { renderScore, destroyAll };
})();
