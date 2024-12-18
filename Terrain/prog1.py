import requests

def get_coordinates(location, api_key):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={api_key}"
    response = requests.get(url).json()
    print("API Response:", response)  # Debug: Print the API response
    if response["status"] == "OK":
        return response["results"][0]["geometry"]["location"]
    else:
        raise ValueError(f"Error fetching coordinates: {response.get('error_message', response['status'])}")
api_key = "AIzaSyCdza-iluhlOHS_aoqUdpXlb1ewFAQ3Q3g"
start_location = "Kolkata"
end_location = "Hyderabad"

start_coords = get_coordinates(start_location, api_key)
end_coords = get_coordinates(end_location, api_key)

print("Start:", start_coords)
print("End:", end_coords)

#----
def get_route(start_coords, end_coords, api_key):
    url = (
        f"https://maps.googleapis.com/maps/api/directions/json?"
        f"origin={start_coords['lat']},{start_coords['lng']}"
        f"&destination={end_coords['lat']},{end_coords['lng']}"
        f"&key={api_key}"
    )
    response = requests.get(url).json()
    if response["status"] == "OK":
        # Extract the polyline points for the route
        return response["routes"][0]["overview_polyline"]["points"]
    else:
        raise ValueError("Error fetching route")
route_polyline = get_route(start_coords, end_coords, api_key)
print("Route polyline:", route_polyline)
#----
def get_map_image(route_polyline, api_key, file_name="map_route.png"):
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"size=600x400"
        f"&maptype=terrain" 
        f"&path=enc:{route_polyline}"
        f"&key={api_key}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, "wb") as file:
            file.write(response.content)
        print(f"Map saved as {file_name}")
    else:
        raise ValueError("Error generating map")
get_map_image(route_polyline, api_key)
#----
from PIL import Image
import matplotlib.pyplot as plt

# Load the map image
map_image = Image.open("map_route.png")

# Display the image
plt.imshow(map_image)
plt.axis("off")
plt.show()

