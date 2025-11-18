# filename: uber_booking_test.py
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

# Set up Chrome options
chrome_options = Options()
# Uncomment the following line if you wish to run Chrome headlessly
# chrome_options.add_argument("--headless")

# Set up the driver with automatic management of the driver executable
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

UBER_URL = "URL_TO_UBER_APP"  # Replace with the URL to the Uber app or website
PICKUP_LOCATION = "Location A"
DESTINATION = "Location B"

try:
    # Navigate to the Uber app
    driver.get(UBER_URL)
    time.sleep(3)  # Wait for the page to load

    # Enter the pickup location
    pickup_input = driver.find_element("name", "pickup")
    pickup_input.clear()
    pickup_input.send_keys(PICKUP_LOCATION)
    time.sleep(2)

    # Enter the destination
    destination_input = driver.find_element("name", "destination")
    destination_input.clear()
    destination_input.send_keys(DESTINATION)
    destination_input.send_keys(Keys.RETURN)

    print("Test executed: Attempted to book a ride from", PICKUP_LOCATION, "to", DESTINATION)

finally:
    time.sleep(5)  # Let user see something before closing
    driver.quit()