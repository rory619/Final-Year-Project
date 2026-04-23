import requests  # For sending HTTP requests to the server

BASE_URL = "http://192.168.0.11:8000"


def test_status():
    """
    Test 1 - Status endpoint
    Sends a GET request to /status and checks the server
    responds with 200 OK and reports itself as online.
    """
    response = requests.get(f"{BASE_URL}/status")
    # Check the server responded without an error
    assert response.status_code == 200, (
        f"Expected status 200 but got {response.status_code}"
    )
    data = response.json()
    # Check the response body contains the expected fields
    assert data["status"] == "online", (
        f"Expected 'online' but got {data['status']}"
    )
    print("PASS - /status returned online")


def test_history_returns_list():
    response = requests.get(f"{BASE_URL}/history")
    # Check the server responded without an error
    assert response.status_code == 200, (
        f"Expected status 200 but got {response.status_code}"
    )
    data = response.json()
    # Check the response is a list
    assert isinstance(data, list), (
        f"Expected a list but got {type(data)}"
    )
    print(f"PASS - /history returned a list with {len(data)} entries")


# Run all tests
if __name__ == "__main__":
    print("Running API tests...\n")
    test_status()
    test_history_returns_list()
    print("\nDone.")
