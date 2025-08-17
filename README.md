# jira-prediction-bot
Design and implement a Chrome Extension that integrates directly with Jira via its API to facilitate real-time human feedback loops on issues, tickets, and task progress.


## 🔹 Plan 1: Jira Resolution Time Prediction Model

### 🎯 Goal:
Develop an AI/ML model that **predicts the time required to resolve Jira issues** based on historical data and current issue characteristics.

### 🔍 Core Tasks:
- **Understand Jira structure and data models**, including:
  - Issue types (Bug, Task, Story, Epic)
  - Status workflows
  - Priority levels
  - Assignee history
  - Timestamps (created, updated, resolved)
  - Comments, labels, and linked issues

### 🧠 Key Features Impacting Resolution Time:
- Issue type and complexity
- Assignee’s historical resolution time
- Number of linked issues or dependencies
- Issue priority and severity
- Team workload and sprint velocity
- Tags/labels (e.g., "backend", "UI", "urgent")
- Number of comments or revisions

### 📈 Model Output:
- Predicted resolution duration (in days/hours)
- Confidence score or uncertainty range
- Explanation of contributing features (feature importance)

---

## 🔹 Plan 2: Chrome Extension with Feedback Loop to the Model

### 🎯 Goal:
Create a **Chrome Extension** that interfaces with Jira and allows **human users to evaluate and give feedback** on the prediction model's accuracy — enabling a **closed-loop learning system**.

### 🧩 Key Features of the Extension:
- Detect when the user is viewing a Jira issue
- Display model's **predicted resolution time**
- Ask user: **"Is this prediction accurate?"** (Yes / No / Needs Adjustment)
- Allow optional feedback text: "Why is this wrong or right?"
- Submit this feedback to a backend or feedback collection service

### 🔄 Feedback Loop Architecture:
- Collect human evaluations and corrections
- Use this feedback to **fine-tune or retrain** the model
- Track agreement rates and prediction improvement over time
- Optionally visualize trends in prediction accuracy across issues or teams

### 🔐 Technical Integration:
- Chrome Extension uses Jira REST API to identify and interact with issues
- Uses secure API to fetch predictions and send feedback
- Model backend stores prediction history + feedback records for training
