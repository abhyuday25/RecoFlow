const state = {
  selectedModel: "hybrid",
  users: [],
  models: [],
};

const userSelect = document.getElementById("userSelect");
const userInput = document.getElementById("userInput");
const modelChips = document.getElementById("modelChips");
const metricsGrid = document.getElementById("metricsGrid");
const resultsGrid = document.getElementById("resultsGrid");
const resultsTitle = document.getElementById("resultsTitle");
const resultsBadge = document.getElementById("resultsBadge");
const statusText = document.getElementById("statusText");
const heroModel = document.getElementById("heroModel");
const recommendButton = document.getElementById("recommendButton");
const shuffleButton = document.getElementById("shuffleButton");

function setStatus(message, tone = "normal") {
  statusText.textContent = message;
  statusText.style.color = tone === "error" ? "#ff9cbc" : "var(--muted)";
}

function renderModels(models) {
  modelChips.innerHTML = "";
  models.forEach((model) => {
    const chip = document.createElement("button");
    chip.className = `chip ${model.id === state.selectedModel ? "active" : ""}`;
    chip.type = "button";
    chip.textContent = model.label;
    chip.addEventListener("click", () => {
      state.selectedModel = model.id;
      heroModel.textContent = model.label;
      renderModels(state.models);
    });
    modelChips.appendChild(chip);
  });
}

function renderUsers(users) {
  userSelect.innerHTML = "";
  users.forEach((userId) => {
    const option = document.createElement("option");
    option.value = String(userId);
    option.textContent = `User ${userId}`;
    userSelect.appendChild(option);
  });

  if (users.length > 0) {
    userSelect.value = String(users[0]);
    userInput.value = String(users[0]);
  }
}

function renderMetrics(metrics) {
  metricsGrid.innerHTML = "";
  if (!metrics.length) {
    metricsGrid.innerHTML = '<div class="empty-state">No evaluation metrics found yet.</div>';
    return;
  }

  metrics.forEach((metric) => {
    const card = document.createElement("article");
    card.className = "metric-card";
    card.innerHTML = `
      <h3>${metric.model_name.replaceAll("_", " ")}</h3>
      <div class="metric-row"><span>RMSE</span><strong>${Number(metric.rmse).toFixed(4)}</strong></div>
      <div class="metric-row"><span>Precision@10</span><strong>${Number(metric.precision_at_k).toFixed(4)}</strong></div>
      <div class="metric-row"><span>Recall@10</span><strong>${Number(metric.recall_at_k).toFixed(4)}</strong></div>
    `;
    metricsGrid.appendChild(card);
  });
}

function renderRecommendations(recommendations, userId) {
  resultsGrid.innerHTML = "";

  if (!recommendations.length) {
    resultsGrid.innerHTML = '<div class="empty-state">No recommendations landed for this request.</div>';
    return;
  }

  recommendations.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "recommendation-card";
    card.innerHTML = `
      <span class="rank-pill">#${index + 1}</span>
      <h3>${item.title}</h3>
      <p class="genre-text">${item.genres}</p>
      <div class="score-row">
        <span>Predicted score</span>
        <strong>${Number(item.score).toFixed(4)}</strong>
      </div>
    `;
    resultsGrid.appendChild(card);
  });

  const activeModel = state.models.find((model) => model.id === state.selectedModel);
  resultsTitle.textContent = `Top picks for User ${userId}`;
  resultsBadge.textContent = activeModel ? activeModel.label : state.selectedModel;
}

async function fetchMeta() {
  const response = await fetch("/meta");
  if (!response.ok) {
    throw new Error("Could not load app metadata.");
  }
  return response.json();
}

async function fetchRecommendations(userId, modelId) {
  const response = await fetch(`/recommend/${userId}?model=${encodeURIComponent(modelId)}`);
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.detail || "Could not fetch recommendations.");
  }

  return payload;
}

async function loadRecommendations(userId) {
  setStatus("Fetching recommendations...");
  recommendButton.disabled = true;

  try {
    const payload = await fetchRecommendations(userId, state.selectedModel);
    renderRecommendations(payload.recommendations, payload.user_id);
    setStatus(`Serving 10 fresh picks for User ${payload.user_id}.`);
  } catch (error) {
    resultsGrid.innerHTML = `<div class="empty-state">${error.message}</div>`;
    resultsTitle.textContent = "Recommendation request failed";
    resultsBadge.textContent = "Try again";
    setStatus(error.message, "error");
  } finally {
    recommendButton.disabled = false;
  }
}

function getResolvedUserId() {
  const typedValue = Number(userInput.value);
  if (Number.isInteger(typedValue) && typedValue > 0) {
    return typedValue;
  }
  return Number(userSelect.value);
}

recommendButton.addEventListener("click", () => {
  const userId = getResolvedUserId();
  if (!userId) {
    setStatus("Pick a valid user id first.", "error");
    return;
  }
  userSelect.value = String(userId);
  userInput.value = String(userId);
  loadRecommendations(userId);
});

shuffleButton.addEventListener("click", () => {
  if (!state.users.length) {
    return;
  }
  const randomUser = state.users[Math.floor(Math.random() * state.users.length)];
  userSelect.value = String(randomUser);
  userInput.value = String(randomUser);
  loadRecommendations(randomUser);
});

userSelect.addEventListener("change", () => {
  userInput.value = userSelect.value;
});

userInput.addEventListener("change", () => {
  if (userInput.value) {
    userSelect.value = userInput.value;
  }
});

async function init() {
  try {
    const meta = await fetchMeta();
    state.users = meta.users || [];
    state.models = meta.models || [];

    renderUsers(state.users);
    renderModels(state.models);
    renderMetrics(meta.metrics || []);

    if (state.users.length) {
      await loadRecommendations(state.users[0]);
    } else {
      setStatus("No users found in the dataset.", "error");
    }
  } catch (error) {
    setStatus(error.message, "error");
    metricsGrid.innerHTML = `<div class="empty-state">${error.message}</div>`;
    resultsGrid.innerHTML = '<div class="empty-state">Frontend loaded, but the backend data call failed.</div>';
  }
}

init();
