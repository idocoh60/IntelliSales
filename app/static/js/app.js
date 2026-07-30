const MONTH_NAMES = ["ינואר","פברואר","מרץ","אפריל","מאי","יוני","יולי","אוגוסט","ספטמבר","אוקטובר","נובמבר","דצמבר"];

let REFERENCE = null;
let charts = {};

function seriesColor(i) {
  return getComputedStyle(document.documentElement).getPropertyValue(`--series-${(i % 4) + 1}`).trim();
}

// ---------- Tabs ----------
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "performance") renderPerformanceTab();
    if (btn.dataset.tab === "insights") renderInsightsTab();
  });
});

// ---------- Predict form ----------
async function loadReference() {
  const res = await fetch("/api/reference");
  REFERENCE = await res.json();

  const catSelect = document.getElementById("customerCategory");
  REFERENCE.customer_categories.forEach(c => {
    const opt = document.createElement("option");
    opt.value = c.CustomerCategoryID;
    opt.textContent = c.CustomerCategoryName;
    catSelect.appendChild(opt);
  });

  const productList = document.getElementById("productList");
  REFERENCE.products.forEach(p => {
    const opt = document.createElement("option");
    opt.value = p.StockItemName;
    opt.dataset.id = p.StockItemID;
    opt.dataset.price = p.RecommendedRetailPrice;
    productList.appendChild(opt);
  });

  const monthSelect = document.getElementById("orderMonth");
  const currentMonth = new Date().getMonth() + 1;
  MONTH_NAMES.forEach((name, idx) => {
    const opt = document.createElement("option");
    opt.value = idx + 1;
    opt.textContent = name;
    if (idx + 1 === currentMonth) opt.selected = true;
    monthSelect.appendChild(opt);
  });
}

document.getElementById("stockItemSearch").addEventListener("input", (e) => {
  const match = [...document.getElementById("productList").options].find(o => o.value === e.target.value);
  if (match) {
    document.getElementById("stockItem").value = match.dataset.id;
    document.getElementById("unitPrice").value = match.dataset.price;
  } else {
    document.getElementById("stockItem").value = "";
  }
});

document.getElementById("predictBtn").addEventListener("click", async () => {
  const errorEl = document.getElementById("formError");
  errorEl.textContent = "";

  const customerCategoryId = document.getElementById("customerCategory").value;
  const stockItemId = document.getElementById("stockItem").value;
  const unitPrice = document.getElementById("unitPrice").value;
  const orderMonth = document.getElementById("orderMonth").value;

  if (!stockItemId) {
    errorEl.textContent = "בחרו מוצר מהרשימה.";
    return;
  }

  const btn = document.getElementById("predictBtn");
  btn.disabled = true;
  btn.textContent = "מחשב...";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        customer_category_id: Number(customerCategoryId),
        stock_item_id: Number(stockItemId),
        unit_price: Number(unitPrice),
        order_month: Number(orderMonth),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "שגיאה לא ידועה");
    renderResult(data);
  } catch (err) {
    errorEl.textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "חשב תחזית";
  }
});

function renderResult(data) {
  document.getElementById("resultCard").hidden = false;

  document.getElementById("probabilityValue").textContent = `${(data.base_probability * 100).toFixed(1)}%`;
  document.getElementById("baselineNote").textContent =
    `ממוצע היסטורי: ${(data.global_baseline * 100).toFixed(1)}%`;
  document.getElementById("quantityValue").textContent = data.predicted_quantity.toFixed(1);

  const tbody = document.getElementById("discountTableBody");
  tbody.innerHTML = "";
  data.discount_table.forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.discount_pct.toFixed(0)}%</td>
      <td>₪${row.simulated_unit_price.toFixed(2)}</td>
      <td>${(row.probability * 100).toFixed(1)}%</td>
      <td><span class="badge ${row.close_deal ? "yes" : "no"}">${row.close_deal ? "לסגור עסקה" : "לא מומלץ"}</span></td>
    `;
    tbody.appendChild(tr);
  });

  const crossSell = document.getElementById("crossSellCard");
  if (data.cross_sell) {
    crossSell.innerHTML = `מוצר מומלץ להצעה נוספת: <strong>${data.cross_sell.StockItemName}</strong>
      <br>נרכש ${data.cross_sell.purchase_count} פעמים בסגמנט זה.`;
  } else {
    crossSell.textContent = "אין המלצת cross-sell זמינה לסגמנט זה.";
  }
}

// ---------- Performance tab ----------
let performanceRendered = false;
async function renderPerformanceTab() {
  if (performanceRendered) return;
  performanceRendered = true;

  const res = await fetch("/api/metrics");
  const m = await res.json();

  document.getElementById("classifierTargetDef").textContent = `יעד: ${m.classifier.target_definition}`;
  document.getElementById("metricAccuracy").textContent = `${(m.classifier.accuracy * 100).toFixed(1)}%`;
  document.getElementById("metricPrecision").textContent = `${(m.classifier.precision * 100).toFixed(1)}%`;
  document.getElementById("metricRecall").textContent = `${(m.classifier.recall * 100).toFixed(1)}%`;
  document.getElementById("metricAuc").textContent = m.classifier.auc.toFixed(3);

  document.getElementById("regressorTargetDef").textContent = `יעד: ${m.regressor.target_definition}`;
  document.getElementById("metricMae").textContent = m.regressor.mae.toFixed(2);
  document.getElementById("metricMeanQty").textContent = m.regressor.mean_quantity.toFixed(1);

  const cm = m.classifier.confusion_matrix;
  charts.confusion = new Chart(document.getElementById("confusionChart"), {
    type: "bar",
    data: {
      labels: ["שלילי אמת", "חיובי אמת"],
      datasets: [
        { label: "חזוי שלילי", data: [cm[0][0], cm[1][0]], backgroundColor: seriesColor(0) },
        { label: "חזוי חיובי", data: [cm[0][1], cm[1][1]], backgroundColor: seriesColor(1) },
      ],
    },
    options: baseChartOptions({ stacked: true }),
  });

  const fi = m.classifier.feature_importance;
  charts.featureImportance = new Chart(document.getElementById("featureImportanceChart"), {
    type: "bar",
    data: {
      labels: Object.keys(fi),
      datasets: [{ label: "חשיבות מאפיין (סיווג)", data: Object.values(fi), backgroundColor: seriesColor(2) }],
    },
    options: baseChartOptions({ indexAxis: "y" }),
  });
}

// ---------- Insights tab ----------
let insightsRendered = false;
async function renderInsightsTab() {
  if (insightsRendered) return;
  insightsRendered = true;

  const res = await fetch("/api/charts");
  const d = await res.json();

  charts.orderValue = new Chart(document.getElementById("orderValueChart"), {
    type: "bar",
    data: {
      labels: d.order_value_distribution.bins.map(b => `${b}`),
      datasets: [{ label: "מספר שורות הזמנה", data: d.order_value_distribution.counts, backgroundColor: seriesColor(0) }],
    },
    options: baseChartOptions({}),
  });

  charts.category = new Chart(document.getElementById("categoryChart"), {
    type: "bar",
    data: {
      labels: d.sales_by_category.labels,
      datasets: [{ label: "סך מכירות (ש״ח)", data: d.sales_by_category.values, backgroundColor: seriesColor(1) }],
    },
    options: baseChartOptions({ indexAxis: "y" }),
  });

  charts.topProducts = new Chart(document.getElementById("topProductsChart"), {
    type: "bar",
    data: {
      labels: d.top_products.labels.map(l => l.length > 28 ? l.slice(0, 28) + "…" : l),
      datasets: [{ label: "סך מכירות (ש״ח)", data: d.top_products.values, backgroundColor: seriesColor(2) }],
    },
    options: baseChartOptions({ indexAxis: "y" }),
  });

  charts.trend = new Chart(document.getElementById("trendChart"), {
    type: "line",
    data: {
      labels: d.sales_trend.labels,
      datasets: [{
        label: "מכירות חודשיות (ש״ח)", data: d.sales_trend.values,
        borderColor: seriesColor(0), backgroundColor: "transparent", tension: 0.25, pointRadius: 0, borderWidth: 2,
      }],
    },
    options: baseChartOptions({}),
  });

  charts.incomeScatter = new Chart(document.getElementById("incomeScatterChart"), {
    type: "scatter",
    data: {
      datasets: [{
        label: "מדינה",
        data: d.sales_vs_income.labels.map((label, i) => ({
          x: d.sales_vs_income.income[i], y: d.sales_vs_income.sales[i], label,
        })),
        backgroundColor: seriesColor(3),
      }],
    },
    options: {
      ...baseChartOptions({}),
      scales: {
        x: { title: { display: true, text: "הכנסה חציונית ($)" }, grid: { color: gridColor() } },
        y: { title: { display: true, text: "סך מכירות (ש״ח)" }, grid: { color: gridColor() } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => `${ctx.raw.label}: $${ctx.raw.x} / ${ctx.raw.y} ש״ח` } },
      },
    },
  });
}

function gridColor() {
  return getComputedStyle(document.documentElement).getPropertyValue("--gridline").trim();
}

function baseChartOptions({ stacked = false, indexAxis = "x" } = {}) {
  const textColor = getComputedStyle(document.documentElement).getPropertyValue("--text-secondary").trim();
  return {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis,
    plugins: {
      legend: { labels: { color: textColor } },
    },
    scales: {
      x: { stacked, ticks: { color: textColor }, grid: { color: gridColor() } },
      y: { stacked, ticks: { color: textColor }, grid: { color: gridColor() } },
    },
  };
}

loadReference();
