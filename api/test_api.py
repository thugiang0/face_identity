import requests


def test_recognize(images_path):
    api_url = "http://localhost:8000/recognize/"
    result = []
    for image_path in images_path:
        with open(image_path, "rb") as image_file:
            files = {"files": (image_path, image_file, "image/jpeg")}
            response = requests.post(api_url, files=files)

        data = response.json()
        result.append(data)
    return result

def test_update(img_path, name):
    api_url = "http://localhost:8000/update_data/"
    with open(img_path, "rb") as image_file:
        files = {"file": (img_path, image_file, "image/jpeg")}
        data = {"name": name}
        response = requests.post(api_url, files=files, data=data)

    result = response.json()
    return result

if __name__ == "__main__":
    # test_recognize
    img_path = ["test/image/blackpink.jpg"]
    result_recognize = test_recognize(img_path)

    test_update
    # img_path = "test/image/Taylor_Swift.jpg"
    # name = "Taylor_Swift"
    # result_update = test_update(img_path, name)


    print(result_recognize)
