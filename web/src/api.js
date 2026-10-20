
const API_URL = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

export const USE_MOCK = import.meta.env.VITE_USE_MOCK === "true";

async function request(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      data?.detail?.detail ||
      data?.detail ||
      data?.message ||
      `API request failed (${response.status})`
    );
  }

  return data;
}

export async function getRepos() {
  const data = await request("/repos");
  return Array.isArray(data) ? data : data.repos ?? [];
}

export async function getConfigs() {
  const data = await request("/configs");
  return Array.isArray(data) ? data : data.configs ?? [];
}

export async function localizeIssue(payload) {
  if (USE_MOCK) {
    const response = await fetch("/mock/localize.json");
    if (!response.ok) throw new Error("Mock localization file not found.");
    return response.json();
  }

  return request("/localize", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getCode({ repo_id, path, start_line, end_line }) {
  const params = new URLSearchParams({
    repo_id,
    path,
    start_line: String(start_line),
    end_line: String(end_line),
  });

  return request(`/code?${params}`);
}

export async function getNeighborhood({
  repo_id,
  entity_id,
  path,
  name,
  hops = 2,
  limit = 60,
}) {
  const params = new URLSearchParams({ repo_id });

  if (entity_id) params.set("entity_id", entity_id);
  if (path) params.set("path", path);
  if (name) params.set("name", name);

  params.set("hops", String(hops));
  params.set("limit", String(limit));

  return request(`/graph/neighborhood?${params}`);
}
