const interactiveArea = document.getElementById("interactiveArea");
const serviceCards = document.querySelectorAll(".service-card");

const diseaseForms = {
  diabetes: [
    { key: "pregnancies", label: "Pregnancies", placeholder: "e.g., 0-10", helper: "Number of prior pregnancies." },
    { key: "glucose", label: "Glucose (mg/dL)", placeholder: "e.g., 70-200 mg/dL", helper: "Current blood glucose concentration." },
    { key: "bp", label: "Blood Pressure (mmHg)", placeholder: "e.g., 60-130 mmHg", helper: "Diastolic blood pressure reading." },
    { key: "skin", label: "Skin Thickness (mm)", placeholder: "e.g., 10-50 mm", helper: "Triceps skin fold thickness." },
    { key: "insulin", label: "Insulin (mu U/ml)", placeholder: "e.g., 15-300", helper: "2-hour serum insulin estimate." },
    { key: "bmi", label: "BMI", placeholder: "e.g., 18.5-40.0", helper: "Body Mass Index based on height and weight." },
    { key: "dpf", label: "Diabetes Pedigree Function", placeholder: "e.g., 0.10-2.50", helper: "Genetic predisposition indicator." },
    { key: "age", label: "Age", placeholder: "e.g., 18-90 years", helper: "Your current age in years." }
  ],
  heart: [
    { key: "age", label: "Age", placeholder: "e.g., 30-85 years", helper: "Age strongly influences heart risk." },
    { key: "trestbps", label: "Resting Blood Pressure (mmHg)", placeholder: "e.g., 90-190 mmHg", helper: "Resting blood pressure level." },
    { key: "chol", label: "Cholesterol (mg/dL)", placeholder: "e.g., 120-320 mg/dL", helper: "Serum cholesterol concentration." },
    { key: "thalach", label: "Max Heart Rate", placeholder: "e.g., 80-210 bpm", helper: "Maximum heart rate achieved." },
    { key: "oldpeak", label: "ST Depression (Oldpeak)", placeholder: "e.g., 0.0-6.0", helper: "ST depression induced by exercise." }
  ],
  cancer: [
    { key: "radius_mean", label: "Radius Mean", placeholder: "e.g., 7.0-30.0", helper: "Average radius from biopsy cell nuclei." },
    { key: "texture_mean", label: "Texture Mean", placeholder: "e.g., 9.0-40.0", helper: "Texture variation of sampled cells." },
    { key: "perimeter_mean", label: "Perimeter Mean", placeholder: "e.g., 40-200", helper: "Average perimeter of cell nuclei." },
    { key: "area_mean", label: "Area Mean", placeholder: "e.g., 150-2500", helper: "Average area of cell nuclei." }
  ]
};

function formFieldHTML(field) {
  return `
        <div class="field-group">
          <label for="${field.key}">${field.label}</label>
          <input type="number" step="any" id="${field.key}" name="${field.key}" placeholder="${field.placeholder}" required />
          <p class="help">${field.helper}</p>
        </div>
      `;
}

function renderPredictionPanel(modelKey, submitLabel) {
  const fields = diseaseForms[modelKey];
  const fieldsHTML = fields.map(formFieldHTML).join("");

  interactiveArea.innerHTML = `
        <form id="predictionForm" action="/predict" method="POST">
          <input type="hidden" name="disease" value="${modelKey}" />
          ${fieldsHTML}
          <div class="btn-row">
            <button type="submit">${submitLabel}</button>
            <a class="link-btn ghost-btn" href="/symptom-checker">Open Symptom Checker</a>
          </div>
        </form>
      `;
}

function renderCancerPanel() {
  const fieldsHTML = diseaseForms.cancer.map(formFieldHTML).join("");
  interactiveArea.innerHTML = `
        <form id="predictionForm" action="/predict" method="POST">
          <input type="hidden" name="disease" value="cancer" />
          ${fieldsHTML}
          <div class="btn-row">
            <button type="submit">Estimate Cancer Risk</button>
          </div>
        </form>
      `;
}

function renderSymptomPanel() {
  interactiveArea.innerHTML = `
        <div class="muted-box">
          Symptom Checker uses guided yes/no questions to suggest likely health conditions.
          Use this when you do not have lab values yet.
        </div>
        <div class="btn-row">
          <a class="link-btn" href="/symptom-checker">Start Symptom Checker</a>
        </div>
      `;
}

function renderHealthRiskPanel() {
  interactiveArea.innerHTML = `
        <form id="healthRiskForm">
          <div class="field-group">
            <label for="hr-age">Age</label>
            <input type="number" id="hr-age" placeholder="e.g., 18-90 years" required />
            <p class="help">Age contributes to baseline health risk.</p>
          </div>
          <div class="field-group">
            <label for="hr-bmi">BMI</label>
            <input type="number" step="0.1" id="hr-bmi" placeholder="e.g., 18.5-40.0" required />
            <p class="help">Body mass index based on weight and height.</p>
          </div>
          <div class="field-group">
            <label for="hr-sleep">Sleep (hours/day)</label>
            <input type="number" step="0.1" id="hr-sleep" placeholder="e.g., 5-9 hours" required />
            <p class="help">Average daily sleep duration.</p>
          </div>
          <div class="field-group">
            <label for="hr-activity">Exercise (days/week)</label>
            <input type="number" id="hr-activity" placeholder="e.g., 0-7 days" required />
            <p class="help">Days with at least 30 mins of moderate activity.</p>
          </div>
          <div class="field-group">
            <label for="hr-smoker">Smoking Status</label>
            <select id="hr-smoker" required>
              <option value="">Select one</option>
              <option value="no">No</option>
              <option value="yes">Yes</option>
            </select>
            <p class="help">Choose "Yes" if currently smoking regularly.</p>
          </div>
          <div class="field-group">
            <label for="hr-bp">Systolic BP (mmHg)</label>
            <input type="number" id="hr-bp" placeholder="e.g., 95-180 mmHg" required />
            <p class="help">Latest systolic blood pressure reading.</p>
          </div>
          <div class="btn-row">
            <button type="submit">Analyze Overall Risk</button>
          </div>
        </form>
        <div id="healthRiskResult"></div>
      `;

  document.getElementById("healthRiskForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const age = Number(document.getElementById("hr-age").value);
    const bmi = Number(document.getElementById("hr-bmi").value);
    const sleep = Number(document.getElementById("hr-sleep").value);
    const activity = Number(document.getElementById("hr-activity").value);
    const smoker = document.getElementById("hr-smoker").value;
    const bp = Number(document.getElementById("hr-bp").value);

    let score = 0;
    if (age >= 50) score += 20; else if (age >= 35) score += 10;
    if (bmi >= 30) score += 20; else if (bmi >= 25) score += 10;
    if (sleep < 6 || sleep > 9) score += 10;
    if (activity < 3) score += 15;
    if (smoker === "yes") score += 20;
    if (bp >= 140) score += 15; else if (bp >= 130) score += 8;

    let label = "Low";
    let className = "ok";
    if (score >= 55) {
      label = "High";
      className = "risk";
    } else if (score >= 30) {
      label = "Moderate";
      className = "warn";
    }

    document.getElementById("healthRiskResult").innerHTML =
      `<span class="${className}">Estimated Overall Risk: ${label} (${score}/100)</span>`;
  });
}

function setActiveService(service) {
  serviceCards.forEach((card) => card.classList.toggle("active", card.dataset.service === service));
  if (service === "heart") renderPredictionPanel("heart", "Predict Heart Disease Risk");
  if (service === "diabetes") renderPredictionPanel("diabetes", "Predict Diabetes Risk");
  if (service === "symptom") renderSymptomPanel();
  if (service === "cancer") renderCancerPanel();
  if (service === "health-risk") renderHealthRiskPanel();
}

serviceCards.forEach((card) => {
  card.addEventListener("click", () => setActiveService(card.dataset.service));
});

setActiveService("heart");

