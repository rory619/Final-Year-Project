import requests  # For sending HTTP requests to the server

BASE_URL = "http://192.168.0.13:8000"


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
<<<<<<< HEAD
    # Test 2 - History endpoint
    # Sends a GET request to /history and checks the server
    # returns a list. Can be empty if no predictions made yet.
=======
>>>>>>> ac90a0daac84861559d2f9372b0cb93666c937e0
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


def test_predict_with_valid_image():
    # Test 3 - Predict endpoint
    # Sends a real image to /predict and checks the response
    # contains a label and a confidence score between 0 and 1.
    with open("/home/rory/fruitchecker/test_images/test.jpg", "rb") as f:
        response = requests.post(
            f"{BASE_URL}/predict",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    # Check the server responded without an error
    assert response.status_code == 200, (
        f"Expected status 200 but got {response.status_code}"
    )
    data = response.json()
    # Check the response contains a label and a score
    assert "label" in data, "Response missing label"
    assert "score" in data, "Response missing score"
    # Check the score is a number between 0 and 1
    assert 0.0 <= data["score"] <= 1.0, (
        f"Score {data['score']} is not between 0 and 1"
    )
    print(f"PASS - /predict returned label='{data['label']}' score={data['score']}")


<<<<<<< HEAD
def test_predict_with_no_file():
    # Test 4 - Error handling
    # Sends a request to /predict with no image attached.
    # The server should return an error, not crash.
    response = requests.post(f"{BASE_URL}/predict")
    assert response.status_code in [400, 422], (
        f"Expected 400 or 422 but got {response.status_code}"
    )
    print(f"PASS - /predict correctly rejected empty request with {response.status_code}")


=======
>>>>>>> ac90a0daac84861559d2f9372b0cb93666c937e0
# Run all tests
if __name__ == "__main__":
    print("Running API tests...\n")
    test_status()
    test_history_returns_list()
    test_predict_with_valid_image()
<<<<<<< HEAD
    test_predict_with_no_file()
=======
>>>>>>> ac90a0daac84861559d2f9372b0cb93666c937e0
    print("\nDone.")
