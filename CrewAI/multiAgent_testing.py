import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew
import os
import json
from datetime import datetime
from langchain_openai.chat_models import AzureChatOpenAI

# Set up OpenAI API
openai_api_key = os.environ["OPENAI_API_KEY"]
#openai_api_key = os.getenv("DIAL_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
api_key = os.getenv("DIAL_API_KEY")
#print(f"Using API Key: {api_key[:4]}****{api_key[-4:]}")


llm = AzureChatOpenAI(
    openai_api_version="2023-07-01-preview",
    api_key=api_key,
    azure_endpoint="https://ai-proxy.lab.epam.com",
    azure_deployment="gpt-35-turbo",
    model="gpt-35-turbo"
)



# AGENTS 

planner = Agent(
    role="Test Planning Strategist",
    goal="Analyze the user story '{user_story}' and create a comprehensive test strategy",
    backstory="You are an experienced Test Manager with expertise in test planning. "
              "You analyze user stories and create detailed test strategies including "
              "test scope, test types needed (functional, non-functional, UI, API), "
              "risk assessment, and testing approach. "
              "You identify what needs to be tested and provide clear direction "
              "to the testing team.",
    allow_delegation=False,
    verbose=True
)

business_analyst = Agent(
    role="Business Analyst",
    goal="Decompose user story '{user_story}' into detailed Acceptance Criteria and Gherkin scenarios",
    backstory="You are a skilled Business Analyst with deep understanding of "
              "requirements engineering and BDD (Behavior Driven Development). "
              "You break down user stories into clear, testable Acceptance Criteria "
              "and write comprehensive Gherkin scenarios (Given-When-Then format) "
              "covering all possible flows including happy paths, alternate paths, "
              "and edge cases.",
    allow_delegation=False,
    verbose=True
)

test_designer = Agent(
    role="Test Case Designer",
    goal="Design comprehensive test cases for all scenarios provided by the Business Analyst",
    backstory="You are a meticulous Test Designer with expertise in creating "
              "detailed, executable test cases. You convert Gherkin scenarios "
              "and Acceptance Criteria into structured test cases with clear "
              "test steps, test data, expected results, and priorities. "
              "You ensure coverage of positive, negative, boundary, and edge cases. "
              "You mark each test case as automatable (Yes/No) based on feasibility.",
    allow_delegation=False,
    verbose=True
)

automation_engineer = Agent(
    role="Automation Test Engineer",
    goal="Write clean, maintainable automation code using Selenium with Python and pytest",
    backstory="You are an expert Automation Engineer proficient in Selenium WebDriver, "
              "Python, and pytest framework. You write Page Object Model (POM) based "
              "automation scripts following coding best practices. "
              "You only automate test cases marked as 'automatable'. "
              "Your code includes proper waits, error handling, and clear assertions. "
              "You use pytest fixtures and follow PEP 8 standards.",
    allow_delegation=False,
    verbose=True
)

code_reviewer = Agent(
    role="Code Review Specialist",
    goal="Review automation code for syntax, compilation, coding standards, and framework-specific best practices",
    backstory="You are a senior Automation Architect who reviews code quality across "
              "multiple languages and frameworks. You adapt your review criteria based on "
              "the technology stack used (e.g., Selenium, Playwright, Cypress, RestAssured, etc.). "
              "You check for syntax errors, language-specific coding standards (PEP 8 for Python, "
              "ESLint for JavaScript, etc.), proper use of framework best practices, "
              "code maintainability, error handling, and test framework conventions. "
              "You provide constructive feedback and approve or request changes. "
              "You DO NOT execute the code, only review it statically.",
    allow_delegation=False,
    verbose=True
)

test_reporter = Agent(
    role="Test Report Analyst",
    goal="Compile all testing artifacts into a comprehensive JSON report",
    backstory="You are a Test Reporting Specialist who consolidates all testing "
              "deliverables into well-structured reports. "
              "You gather test strategy, acceptance criteria, test cases, "
              "automation code, and code review feedback, then format everything "
              "into a clear JSON structure for easy consumption and tracking.",
    allow_delegation=False,
    verbose=True
)


# TASKS DEFINITION 

task_planning = Task(
    description=(
        "Analyze the user story: '{user_story}'\n\n"
        "Create a comprehensive test strategy that includes:\n"
        "1. Test scope and objectives\n"
        "2. Types of testing required (functional, UI, API, etc.)\n"
        "3. Risk assessment and mitigation\n"
        "4. Testing approach and priorities\n"
        "5. Entry and exit criteria\n"
        "6. Test environment requirements\n"
        "7. Assumptions and dependencies"
    ),
    expected_output="A detailed test strategy document covering all aspects of testing "
                   "for the given user story with clear priorities and approach.",
    agent=planner,
)

task_ba_analysis = Task(
    description=(
        "Based on the test strategy and user story: '{user_story}'\n\n"
        "Create:\n"
        "1. Detailed Acceptance Criteria (AC) - minimum 5 ACs\n"
        "2. Gherkin scenarios in Given-When-Then format\n"
        "3. Cover happy path, alternate flows, negative scenarios, and edge cases\n"
        "4. Include data variations and boundary conditions\n"
        "5. Ensure scenarios are testable and specific"
    ),
    expected_output="A comprehensive list of Acceptance Criteria and Gherkin scenarios "
                   "in proper BDD format, covering all possible test scenarios.",
    agent=business_analyst,
)

task_test_design = Task(
    description=(
        "Based on the Acceptance Criteria and Gherkin scenarios provided:\n\n"
        "Design detailed test cases in the following structure:\n"
        "- TC_ID: Unique identifier\n"
        "- Requirement_Mapping: Which AC/scenario it maps to\n"
        "- Test_Description: Clear description\n"
        "- Test_Steps: Detailed step-by-step actions\n"
        "- Test_Type: Positive/Negative/Boundary/Edge\n"
        "- Priority: Critical/High/Medium/Low\n"
        "- Test_Data: Specific test data to use\n"
        "- Expected_Result: Clear expected outcome\n"
        "- Automatable: Yes/No with justification\n\n"
        "Create at least 10-15 comprehensive test cases covering all scenarios."
    ),
    expected_output="A comprehensive set of test cases in tabular/structured format "
                   "with all required fields filled, ready for execution.",
    agent=test_designer,
)

task_automation = Task(
    description=(
        "Based on the test cases marked as 'Automatable: Yes':\n\n"
        "Write Selenium Python automation code with:\n"
        "1. Page Object Model (POM) structure\n"
        "2. pytest framework with fixtures\n"
        "3. Explicit waits (WebDriverWait)\n"
        "4. Clear assertions using pytest assertions\n"
        "5. Proper exception handling\n"
        "6. Comments and docstrings\n"
        "7. Follow PEP 8 coding standards\n"
        "8. Configuration management (base URL, timeouts)\n\n"
        "Include:\n"
        "- conftest.py for fixtures\n"
        "- Page Object classes\n"
        "- Test classes with test methods\n"
        "- requirements.txt"
    ),
    expected_output="Complete, production-ready Selenium Python automation code "
                   "following best practices, organized in proper structure with "
                   "POM pattern and pytest framework.",
    agent=automation_engineer,
)

task_code_review = Task(
    description=(
        "Review the automation code provided by the Automation Engineer:\n"
        "Identify the language and framework used, then check for:\n"
        "1. Syntax correctness and potential compilation errors\n"
        "2. Language-specific coding standards (e.g., PEP 8 for Python, ESLint for JavaScript, etc.)\n"
        "3. Framework-specific best practices:\n"
        "   - For web automation: proper waits, robust locators, page object pattern\n"
        "   - For API testing: proper assertions, response validation, error handling\n"
        "   - For mobile: proper app interactions, device handling\n"
        "4. Test framework conventions (pytest, JUnit, TestNG, Mocha, etc.)\n"
        "5. Code maintainability and readability\n"
        "6. Error handling and edge cases\n"
        "7. Code comments and documentation\n"
        "8. Security considerations (no hardcoded credentials, sensitive data)\n\n"
        "Provide:\n"
        "- Technology stack identified\n"
        "- Overall assessment (Approved/Changes Required)\n"
        "- List of issues found (if any) with severity (Critical/Major/Minor)\n"
        "- Recommendations for improvement\n"
        "- Code quality score (1-10)"
    ),
    expected_output="A detailed code review report with technology stack identification, "
                   "assessment, issues identified with severity levels, recommendations, "
                   "and approval status.",
    agent=code_reviewer,
)

task_reporting = Task(
    description=(
        "Compile all testing artifacts into a comprehensive JSON report:\n\n"
        "Include:\n"
        "1. Test Strategy summary\n"
        "2. Acceptance Criteria and Scenarios\n"
        "3. All Test Cases\n"
        "4. Automation Code (if applicable)\n"
        "5. Code Review Results\n"
        "6. Summary statistics (total tests, automatable count, priorities breakdown)\n"
        "7. Metadata (timestamp, user story, iteration)\n\n"
        "Format: Well-structured JSON with proper nesting and readability." 
        "Write only the JSON output without any extra explanations or comments or delimiters like ```."
        "The extension of the file should be .json"
    ),
    expected_output="A comprehensive JSON report containing all testing deliverables "
                   "with proper structure and formatting, ready to be saved to file.",
    agent=test_reporter,
)


# CREW SETUP 

def run_testing_crew(user_story, iteration_name="Sprint-1", enable_human_input=False):
    """
    Execute the testing crew for a given user story.
    
    Parameters:
    - user_story: The user story to test
    - iteration_name: Name of the sprint/iteration
    - enable_human_input: Flag to enable human review (True/False)
    """
    
    # Update human input mode for agents if needed
    if enable_human_input:
        planner.allow_delegation = True
        business_analyst.allow_delegation = True
    
    # Create the crew
    testing_crew = Crew(
        agents=[
            planner,
            business_analyst,
            test_designer,
            automation_engineer,
            code_reviewer,
            test_reporter
        ],
        tasks=[
            task_planning,
            task_ba_analysis,
            task_test_design,
            task_automation,
            task_code_review,
            task_reporting
        ],
        verbose=True
    )
    
    # Execute the crew
    print(f"\n############################")
    print(f"Starting Testing Activities for: {iteration_name}")
    print(f"User Story: {user_story}")
    print(f"##############################\n")
    
    result = testing_crew.kickoff(inputs={
        "user_story": user_story,
        "iteration": iteration_name
    })
    
    # Save to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"testing_report_{iteration_name}_{timestamp}.json"
    
    try:
        # Try to parse result as JSON, 
        # if not, create structured output
        if isinstance(result, str):
            output_data = {
                "metadata": {
                    "iteration": iteration_name,
                    "user_story": user_story,
                    "timestamp": timestamp,
                    "execution_date": datetime.now().isoformat()
                },
                "report": result
            }
        else:
            output_data = result
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n############################")
        print(f"Report saved to: {filename}")
        print(f"##############################\n")
        
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        # Fallback: save as text
        with open(filename.replace('.json', '.txt'), 'w', encoding='utf-8') as f:
            f.write(str(result))
    
    return result


# MAIN EXECUTION 

if __name__ == "__main__":
    
    # Example 1: Uber ride booking
    user_story_1 = "As a rider, I want to book a ride from location A to location B using the Uber app, so that I can reach my destination conveniently"
    
    result = run_testing_crew(
        user_story=user_story_1,
        iteration_name="Sprint-1",
        enable_human_input=True
    )
    
    print("\n############################")
    print("TESTING CREW EXECUTION COMPLETED")
    print("##############################")
    print("\nResult Below:")
    print(str(result)[:500] + "..." if len(str(result)) > 500 else str(result))
    
    