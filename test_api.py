import requests  # For sending HTTP requests to the server

BASE_URL = "http://192.168.0.13:8000"


def test_status():
    # Test 1 - Status endpoint
    # Sends a GET request to /status and checks the server
    # responds with 200 OK and reports itself as online.
    response = requests.get(f"{BASE_URL}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    print("PASS - /status returned online")


def test_history_returns_list():
    
    # Test 2 - History endpoint
    # Sends a GET request to /history and checks the server
    # returns a list. Can be empty if no predictions made yet.

    

    response = requests.get(f"{BASE_URL}/history")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    print(f"PASS - /history returned a list with {len(data)} entries")


def test_predict_with_valid_image():
    
    # Test 3 - Predict endpoint
    # Sends image to /predict and checks the response
    # contains a label and a confidence score between 0 and 1.
    with open("/home/rory/fruitchecker/test_images/test.jpg", "rb") as f:
        response = requests.post(
            f"{BASE_URL}/predict",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "score" in data
    assert 0.0 <= data["score"] <= 1.0
    print(f"PASS - /predict returned label='{data['label']}' score={data['score']}")


def test_predict_with_no_file():
    
    # Test 4 - Error handling
    # Sends a request to /predict with no image attached.
    # The server should return an error.
    response = requests.post(f"{BASE_URL}/predict")
    assert response.status_code in [400, 422]
    print(f"PASS - /predict correctly rejected empty request with {response.status_code}")



def test_history_saves_after_prediction():
    
    # Test 5 - Integration test
    # Makes a prediction then checks /history to confirm
    # the result was saved by the server.
    with open("/home/rory/fruitchecker/test_images/test.jpg", "rb") as f:
        requests.post(f"{BASE_URL}/predict",files={"file": ("test.jpg", f, "image/jpeg")} )
    response = requests.get(f"{BASE_URL}/history")
    data = response.json()
    assert len(data) > 0
    latest = data[0]
    assert "label" in latest
    assert "score" in latest
    assert "timestamp" in latest
    print(f"PASS - history saved entry with label='{latest['label']}' score={latest['score']}")


# Run all tests
if __name__ == "__main__":
    print("Running API tests...\n")
    test_status()
    test_history_returns_list()
    test_predict_with_valid_image()
    test_predict_with_no_file()

    test_history_saves_after_prediction()
    print("\nDone.")
