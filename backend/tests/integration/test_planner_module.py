from fastapi.testclient import TestClient


def _register_and_headers(client: TestClient, email: str) -> dict[str, str]:
    res = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "StrongPassword123!"},
    )
    assert res.status_code == 201
    token = res.json()["data"]["tokens"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_planner_tasks_crud_and_user_isolation(client: TestClient) -> None:
    headers_a = _register_and_headers(client, "planner_a@example.com")
    headers_b = _register_and_headers(client, "planner_b@example.com")

    # Create task for user A
    create_res = client.post(
        "/api/v1/planner/tasks",
        headers=headers_a,
        json={
            "title": "Finish Calculus Assignment",
            "category": "Study",
            "date": "2026-11-14",
            "time_label": "10:00 AM",
            "duration_minutes": 60,
        },
    )
    assert create_res.status_code == 201
    created = create_res.json()["data"]
    assert created["title"] == "Finish Calculus Assignment"
    assert created["category"] == "Study"
    assert created["task_date"] == "2026-11-14"
    assert created["completed"] is False
    task_id = created["id"]

    # List tasks for user A
    list_a = client.get("/api/v1/planner/tasks", headers=headers_a)
    assert list_a.status_code == 200
    assert list_a.json()["data"]["total"] == 1

    # User B cannot see or mutate user A's task
    list_b = client.get("/api/v1/planner/tasks", headers=headers_b)
    assert list_b.status_code == 200
    assert list_b.json()["data"]["total"] == 0

    patch_b = client.patch(
        f"/api/v1/planner/tasks/{task_id}",
        headers=headers_b,
        json={"completed": True},
    )
    assert patch_b.status_code == 404

    # User A toggles completion
    patch_a = client.patch(
        f"/api/v1/planner/tasks/{task_id}",
        headers=headers_a,
        json={"completed": True},
    )
    assert patch_a.status_code == 200
    assert patch_a.json()["data"]["completed"] is True

    # User A deletes task
    del_a = client.delete(f"/api/v1/planner/tasks/{task_id}", headers=headers_a)
    assert del_a.status_code == 200
    assert del_a.json()["data"]["deleted"] is True

    list_after = client.get("/api/v1/planner/tasks", headers=headers_a)
    assert list_after.status_code == 200
    assert list_after.json()["data"]["total"] == 0

