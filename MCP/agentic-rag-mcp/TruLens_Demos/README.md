# Email Confirmation Test Automation

This project contains Selenium WebDriver automation tests for testing the Email Confirmation requirement.

## Test Framework
- **Language:** Java
- **Testing Framework:** TestNG
- **Web Automation:** Selenium WebDriver
- **Build Tool:** Maven
- **Email Verification:** JavaMail API

## Test Requirements
- **Feature:** Email confirmation to user
- **Requirement:** User should receive an email confirmation for order confirmation
- **Context:** E-Store product features for making Marvel Electronics and Home Entertainment project live

## Test Coverage

### 1. testSuccessfulOrderEmailConfirmation
- **Description:** Tests successful order placement and email confirmation
- **Steps:**
  1. Login to E-Store application
  2. Browse and add product to cart
  3. Proceed to checkout
  4. Fill shipping and payment details
  5. Place order
  6. Verify order confirmation page
  7. Verify email confirmation received

### 2. testEmailContentValidation  
- **Description:** Validates email content contains required information
- **Validation Points:**
  - Company name (Marvel Electronics)
  - Order confirmation text
  - Order number
  - Thank you message
  - Order details section
  - Shipping address
  - Total amount
  - Contact information

### 3. testNoDuplicateEmailConfirmations
- **Description:** Ensures no duplicate email confirmations are sent
- **Validation:** Exactly one email per order

### 4. testEmailConfirmationMultipleProducts
- **Description:** Tests email confirmation for different product categories
- **Data Provider:** Tests multiple product types from Electronics and Home Entertainment

## Prerequisites

### 1. Java Development Kit (JDK)
- Java 11 or higher installed

### 2. Maven
- Apache Maven 3.6+ installed

### 3. WebDriver Setup
- Chrome browser installed
- Firefox browser installed (optional)
- WebDriverManager automatically handles driver binaries

### 4. Email Configuration
- Gmail account for testing
- App password enabled for Gmail
- Update email credentials in test class:
  ```java
  private static final String TEST_EMAIL = "your-test-email@gmail.com";
  private static final String EMAIL_PASSWORD = "your-app-password";
  ```

## Project Structure
```
src/
├── main/java/
│   └── com/marvelelectronics/
│       ├── pages/          # Page Object classes
│       └── utils/          # Utility classes
├── test/java/
│   └── com/marvelelectronics/tests/
│       └── EmailConfirmationTest.java
├── test/resources/
│   ├── testdata/           # Test data files
│   └── config/             # Configuration files
├── pom.xml                 # Maven configuration
└── testng.xml             # TestNG suite configuration
```

## How to Run Tests

### 1. Install Dependencies
```bash
mvn clean install
```

### 2. Run All Tests
```bash
mvn test
```

### 3. Run with Specific Browser
```bash
mvn test -Dbrowser=chrome
mvn test -Dbrowser=firefox
```

### 4. Run Specific Test Method
```bash
mvn test -Dtest=EmailConfirmationTest#testSuccessfulOrderEmailConfirmation
```

### 5. Run with TestNG XML
```bash
mvn test -DsuiteXmlFile=testng.xml
```

### 6. Run with Maven Profiles
```bash
mvn test -Pchrome
mvn test -Pfirefox
```

## Test Configuration

### Browser Configuration
- Default: Chrome
- Supported: Chrome, Firefox
- Configurable via TestNG parameter or system property

### Email Configuration
- IMAP server: imap.gmail.com
- Port: 993 (SSL enabled)
- Protocol: IMAP
- Authentication: Username/Password (App Password for Gmail)

### Timeouts
- Implicit wait: 10 seconds
- Explicit wait: 30 seconds
- Page load timeout: 30 seconds
- Email processing wait: 30 seconds

## Test Data
The tests use a DataProvider for testing multiple scenarios:
- Electronics products: Smartphone, Laptop, Tablet
- Home Entertainment products: Smart TV, Sound System
- Different customer names for shipping

## Reporting
- TestNG HTML reports generated in `target/surefire-reports/`
- ExtentReports can be integrated for enhanced reporting
- Console logging for test execution details

## Email Verification Process
1. Tests place orders on the E-Store platform
2. Wait for email processing (30 seconds)
3. Connect to Gmail IMAP server
4. Search for emails from Marvel Electronics
5. Verify email content contains order number
6. Validate email structure and required information

## Troubleshooting

### Common Issues:
1. **Email not received:** Check spam folder, verify email credentials
2. **WebDriver issues:** Ensure browsers are installed and updated
3. **Element not found:** Verify web application is accessible
4. **Timeout errors:** Increase wait times if needed

### Configuration Updates:
- Update base URL in test class
- Modify email settings for different providers
- Adjust timeout values based on application performance

## Best Practices Implemented
- Page Object Model pattern
- Explicit waits for better reliability
- Data-driven testing with TestNG DataProvider
- Proper test isolation and cleanup
- Comprehensive error handling
- Detailed logging and reporting
- Cross-browser testing support
- Maven profiles for different environments

## Dependencies Used
- Selenium WebDriver 4.15.0
- TestNG 7.8.0
- WebDriverManager 5.6.2
- JavaMail API 1.6.2
- ExtentReports 5.0.9
- Log4j 2.21.1
- Apache Commons Email 1.5
- Jackson JSON 2.15.2
