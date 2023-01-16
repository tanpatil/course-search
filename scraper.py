from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time
import csv

driver = webdriver.Firefox()
try:
	driver.get("https://catalog.ucsd.edu/front/courses.html")
	elems = driver.find_elements(By.LINK_TEXT, "courses")
	symbs = []
	for elem in elems: 
		link = elem.get_attribute("href")
		symbs.append(link.split('/')[-1])
	
	print(symbs)
		

	with open('courses.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow(['name', 'description'])
		
		for symb in symbs:
			link = f"https://catalog.ucsd.edu/courses/{symb}"
			driver.get(link)
			# find all course names 
			names = driver.find_elements(By.CLASS_NAME, "course-name")

			# find all course descriptions
			descs = driver.find_elements(By.CLASS_NAME, "course-descriptions")
			for i in range(len(names)):
				if i >= len(descs):
					break
				print("name:", names[i].text)
				print("description:", descs[i].text)
				row = [names[i].text, descs[i].text]
				writer.writerow(row)

	elems.clear()
finally:
	driver.close()