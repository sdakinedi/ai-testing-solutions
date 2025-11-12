# filename: automate_uber_ride_booking.py

# Import necessary libraries
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

# Create an instance of Chrome WebDriver
driver = webdriver.Chrome()

# Open the Uber website
driver.get("https://www.uber.com")

# Locate the "Where to?" input field and enter location A
start_location = driver.find_element_by_name("destination")
start_location.send_keys("Location A")

# Locate the "Where to?" input field and enter location B
end_location = driver.find_element_by_name("destination")
end_location.send_keys("Location B")

# Click on the "Search" button to find a ride
search_button = driver.find_element_by_id("search-button")
search_button.click()

# Add a wait time to see the results
time.sleep(5)

# Close the browser window
driver.quit()