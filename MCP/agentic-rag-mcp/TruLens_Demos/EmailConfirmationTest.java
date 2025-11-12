package com.marvelelectronics.tests;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.By;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.Select;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;
import org.testng.annotations.Parameters;
import org.testng.annotations.DataProvider;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.AfterMethod;

import javax.mail.*;
import javax.mail.search.SearchTerm;
import javax.mail.search.SubjectTerm;
import javax.mail.search.AndTerm;
import javax.mail.search.FromTerm;
import javax.mail.internet.InternetAddress;
import java.util.Properties;
import java.util.concurrent.TimeUnit;
import java.time.Duration;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

/**
 * EmailConfirmationTest - Test automation for Email Confirmation requirement
 * 
 * Feature: Email confirmation to user
 * Requirement: User should receive an email confirmation for order confirmation
 * Context: E-Store product features for making Marvel Electronics and Home Entertainment project live
 * 
 * Test Coverage:
 * - Successful order placement and email confirmation
 * - Email content validation 
 * - No duplicate emails
 * - Email delivery timing
 */
public class EmailConfirmationTest {
    
    private WebDriver driver;
    private WebDriverWait wait;
    
    // Test Configuration
    private static final String ESTORE_BASE_URL = "https://marvelelectronics.com";
    private static final String TEST_EMAIL = "testuser@gmail.com";
    private static final String TEST_PASSWORD = "TestPass123!";
    private static final String EMAIL_PASSWORD = "app_password"; // Gmail app password
    private static final int TIMEOUT_SECONDS = 30;
    
    // Email Configuration
    private static final String MAIL_HOST = "imap.gmail.com";
    private static final String MAIL_PORT = "993";
    private static final String EXPECTED_SENDER = "noreply@marvelelectronics.com";
    
    @BeforeClass
    @Parameters({"browser"})
    public void setUp(String browser) {
        System.out.println("Setting up test environment...");
        
        // Initialize WebDriver based on browser parameter
        if (browser.equalsIgnoreCase("chrome")) {
            // Using WebDriverManager for automatic driver management
            io.github.bonigarcia.wdm.WebDriverManager.chromedriver().setup();
            driver = new ChromeDriver();
        } else if (browser.equalsIgnoreCase("firefox")) {
            io.github.bonigarcia.wdm.WebDriverManager.firefoxdriver().setup();
            driver = new FirefoxDriver();
        } else {
            throw new IllegalArgumentException("Browser not supported: " + browser);
        }
        
        // Configure WebDriver
        driver.manage().window().maximize();
        driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(10));
        driver.manage().timeouts().pageLoadTimeout(Duration.ofSeconds(30));
        
        // Initialize WebDriverWait
        wait = new WebDriverWait(driver, Duration.ofSeconds(TIMEOUT_SECONDS));
        
        System.out.println("WebDriver setup completed for browser: " + browser);
    }
    
    @BeforeMethod
    public void navigateToHomePage() {
        driver.get(ESTORE_BASE_URL);
        System.out.println("Navigated to E-Store homepage");
    }
    
    @Test(priority = 1, description = "Test successful order placement and email confirmation")
    public void testSuccessfulOrderEmailConfirmation() {
        System.out.println("Starting test: testSuccessfulOrderEmailConfirmation");
        
        try {
            // Step 1: Login to the application
            loginToApplication();
            
            // Step 2: Browse and add product to cart
            String productName = addProductToCart("Electronics", "Smartphone Galaxy S24");
            
            // Step 3: Proceed to checkout
            proceedToCheckout();
            
            // Step 4: Fill order information
            fillShippingDetails();
            fillPaymentDetails();
            
            // Step 5: Place the order
            String orderNumber = placeOrder();
            Assert.assertNotNull(orderNumber, "Order number should not be null");
            System.out.println("Order placed successfully. Order Number: " + orderNumber);
            
            // Step 6: Verify order confirmation page
            verifyOrderConfirmationPage(orderNumber);
            
            // Step 7: Wait for email processing
            waitForEmailProcessing();
            
            // Step 8: Verify email confirmation
            boolean emailReceived = verifyEmailConfirmation(orderNumber);
            Assert.assertTrue(emailReceived, "Email confirmation was not received for order: " + orderNumber);
            
            System.out.println("Test completed successfully: Email confirmation received for order " + orderNumber);
            
        } catch (Exception e) {
            System.err.println("Test failed with exception: " + e.getMessage());
            e.printStackTrace();
            Assert.fail("Test failed due to exception: " + e.getMessage());
        }
    }
    
    @Test(priority = 2, description = "Test email content validation")
    public void testEmailContentValidation() {
        System.out.println("Starting test: testEmailContentValidation");
        
        try {
            // Complete order flow
            loginToApplication();
            addProductToCart("Electronics", "Laptop Pro 16");
            proceedToCheckout();
            fillShippingDetails();
            fillPaymentDetails();
            String orderNumber = placeOrder();
            
            // Wait for email
            waitForEmailProcessing();
            
            // Get email content
            String emailContent = getEmailContent(orderNumber);
            Assert.assertNotNull(emailContent, "Email content should not be null");
            
            // Validate email content
            validateEmailContent(emailContent, orderNumber);
            
            System.out.println("Test completed successfully: Email content validation passed");
            
        } catch (Exception e) {
            System.err.println("Test failed with exception: " + e.getMessage());
            Assert.fail("Test failed due to exception: " + e.getMessage());
        }
    }
    
    @Test(priority = 3, description = "Test no duplicate email confirmations")
    public void testNoDuplicateEmailConfirmations() {
        System.out.println("Starting test: testNoDuplicateEmailConfirmations");
        
        try {
            // Complete order flow
            loginToApplication();
            addProductToCart("Electronics", "Tablet Air 12");
            proceedToCheckout();
            fillShippingDetails();
            fillPaymentDetails();
            String orderNumber = placeOrder();
            
            // Wait longer to ensure all emails are received
            Thread.sleep(60000); // Wait 1 minute
            
            // Check email count
            int emailCount = getEmailCount(orderNumber);
            Assert.assertEquals(emailCount, 1, "Expected exactly 1 email, but found " + emailCount + " emails for order " + orderNumber);
            
            System.out.println("Test completed successfully: No duplicate emails found");
            
        } catch (Exception e) {
            System.err.println("Test failed with exception: " + e.getMessage());
            Assert.fail("Test failed due to exception: " + e.getMessage());
        }
    }
    
    @Test(priority = 4, dataProvider = "orderTestData", description = "Test email confirmation for multiple product types")
    public void testEmailConfirmationMultipleProducts(String category, String productName, String customerName) {
        System.out.println("Starting test: testEmailConfirmationMultipleProducts for " + productName);
        
        try {
            loginToApplication();
            addProductToCart(category, productName);
            proceedToCheckout();
            fillShippingDetails(customerName);
            fillPaymentDetails();
            String orderNumber = placeOrder();
            
            waitForEmailProcessing();
            
            boolean emailReceived = verifyEmailConfirmation(orderNumber);
            Assert.assertTrue(emailReceived, "Email confirmation not received for " + productName);
            
            System.out.println("Test completed successfully for product: " + productName);
            
        } catch (Exception e) {
            System.err.println("Test failed for product " + productName + ": " + e.getMessage());
            Assert.fail("Test failed for product " + productName + ": " + e.getMessage());
        }
    }
    
    // Helper Methods
    
    private void loginToApplication() {
        System.out.println("Logging into application...");
        
        WebElement loginButton = wait.until(ExpectedConditions.elementToBeClickable(By.id("login-btn")));
        loginButton.click();
        
        WebElement emailField = wait.until(ExpectedConditions.presenceOfElementLocated(By.id("email")));
        emailField.clear();
        emailField.sendKeys(TEST_EMAIL);
        
        WebElement passwordField = driver.findElement(By.id("password"));
        passwordField.clear();
        passwordField.sendKeys(TEST_PASSWORD);
        
        WebElement submitButton = driver.findElement(By.id("submit-login"));
        submitButton.click();
        
        // Wait for successful login
        wait.until(ExpectedConditions.presenceOfElementLocated(By.className("user-dashboard")));
        System.out.println("Successfully logged into application");
    }
    
    private String addProductToCart(String category, String productName) {
        System.out.println("Adding product to cart: " + productName);
        
        // Navigate to category
        WebElement categoryLink = wait.until(ExpectedConditions.elementToBeClickable(By.linkText(category)));
        categoryLink.click();
        
        // Find and click on specific product
        WebElement product = wait.until(ExpectedConditions.elementToBeClickable(
            By.xpath("//div[@class='product-item']//h3[contains(text(),'" + productName + "')]")));
        product.click();
        
        // Add to cart
        WebElement addToCartBtn = wait.until(ExpectedConditions.elementToBeClickable(By.id("add-to-cart")));
        addToCartBtn.click();
        
        // Wait for cart confirmation
        wait.until(ExpectedConditions.presenceOfElementLocated(By.className("cart-success-message")));
        System.out.println("Product added to cart successfully: " + productName);
        
        return productName;
    }
    
    private void proceedToCheckout() {
        System.out.println("Proceeding to checkout...");
        
        WebElement cartIcon = wait.until(ExpectedConditions.elementToBeClickable(By.id("cart-icon")));
        cartIcon.click();
        
        WebElement checkoutBtn = wait.until(ExpectedConditions.elementToBeClickable(By.id("checkout-btn")));
        checkoutBtn.click();
        
        // Wait for checkout page to load
        wait.until(ExpectedConditions.presenceOfElementLocated(By.className("checkout-form")));
        System.out.println("Reached checkout page");
    }
    
    private void fillShippingDetails() {
        fillShippingDetails("John Doe");
    }
    
    private void fillShippingDetails(String customerName) {
        System.out.println("Filling shipping details for: " + customerName);
        
        String[] names = customerName.split(" ");
        String firstName = names[0];
        String lastName = names.length > 1 ? names[1] : "Doe";
        
        WebElement firstNameField = wait.until(ExpectedConditions.presenceOfElementLocated(By.id("firstName")));
        firstNameField.clear();
        firstNameField.sendKeys(firstName);
        
        WebElement lastNameField = driver.findElement(By.id("lastName"));
        lastNameField.clear();
        lastNameField.sendKeys(lastName);
        
        WebElement addressField = driver.findElement(By.id("address"));
        addressField.clear();
        addressField.sendKeys("123 Test Street");
        
        WebElement cityField = driver.findElement(By.id("city"));
        cityField.clear();
        cityField.sendKeys("Test City");
        
        WebElement zipField = driver.findElement(By.id("zipCode"));
        zipField.clear();
        zipField.sendKeys("12345");
        
        Select stateSelect = new Select(driver.findElement(By.id("state")));
        stateSelect.selectByValue("CA");
        
        WebElement phoneField = driver.findElement(By.id("phone"));
        phoneField.clear();
        phoneField.sendKeys("555-123-4567");
        
        System.out.println("Shipping details filled successfully");
    }
    
    private void fillPaymentDetails() {
        System.out.println("Filling payment details...");
        
        WebElement cardNumberField = wait.until(ExpectedConditions.presenceOfElementLocated(By.id("cardNumber")));
        cardNumberField.clear();
        cardNumberField.sendKeys("4111111111111111");
        
        WebElement expiryField = driver.findElement(By.id("expiryDate"));
        expiryField.clear();
        expiryField.sendKeys("12/25");
        
        WebElement cvvField = driver.findElement(By.id("cvv"));
        cvvField.clear();
        cvvField.sendKeys("123");
        
        WebElement cardNameField = driver.findElement(By.id("cardName"));
        cardNameField.clear();
        cardNameField.sendKeys("John Doe");
        
        System.out.println("Payment details filled successfully");
    }
    
    private String placeOrder() {
        System.out.println("Placing order...");
        
        WebElement placeOrderBtn = wait.until(ExpectedConditions.elementToBeClickable(By.id("place-order-btn")));
        placeOrderBtn.click();
        
        // Wait for order confirmation page and extract order number
        WebElement orderNumberElement = wait.until(ExpectedConditions.presenceOfElementLocated(By.id("order-number")));
        String orderNumber = orderNumberElement.getText().trim();
        
        System.out.println("Order placed successfully with order number: " + orderNumber);
        return orderNumber;
    }
    
    private void verifyOrderConfirmationPage(String orderNumber) {
        System.out.println("Verifying order confirmation page...");
        
        WebElement confirmationMessage = wait.until(ExpectedConditions.presenceOfElementLocated(By.className("order-confirmation")));
        Assert.assertTrue(confirmationMessage.isDisplayed(), "Order confirmation message should be displayed");
        
        String confirmationText = confirmationMessage.getText();
        Assert.assertTrue(confirmationText.contains(orderNumber), "Order confirmation should contain order number: " + orderNumber);
        Assert.assertTrue(confirmationText.contains("Thank you"), "Order confirmation should contain thank you message");
        
        System.out.println("Order confirmation page verified successfully");
    }
    
    private void waitForEmailProcessing() {
        System.out.println("Waiting for email processing...");
        try {
            Thread.sleep(30000); // Wait 30 seconds for email processing
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    
    private boolean verifyEmailConfirmation(String orderNumber) {
        System.out.println("Verifying email confirmation for order: " + orderNumber);
        
        try {
            Properties props = new Properties();
            props.put("mail.store.protocol", "imap");
            props.put("mail.imap.host", MAIL_HOST);
            props.put("mail.imap.port", MAIL_PORT);
            props.put("mail.imap.ssl.enable", "true");
            
            Session session = Session.getDefaultInstance(props);
            Store store = session.getStore("imap");
            store.connect(MAIL_HOST, TEST_EMAIL, EMAIL_PASSWORD);
            
            Folder folder = store.getFolder("INBOX");
            folder.open(Folder.READ_ONLY);
            
            // Search for emails from Marvel Electronics about this order
            SearchTerm subjectTerm = new SubjectTerm("Order Confirmation");
            SearchTerm fromTerm = new FromTerm(new InternetAddress(EXPECTED_SENDER));
            SearchTerm combinedTerm = new AndTerm(subjectTerm, fromTerm);
            
            Message[] messages = folder.search(combinedTerm);
            
            boolean emailFound = false;
            for (Message msg : messages) {
                String content = getTextContent(msg);
                if (content.contains(orderNumber)) {
                    emailFound = true;
                    System.out.println("Email confirmation found for order: " + orderNumber);
                    break;
                }
            }
            
            folder.close(false);
            store.close();
            
            return emailFound;
            
        } catch (Exception e) {
            System.err.println("Error verifying email confirmation: " + e.getMessage());
            return false;
        }
    }
    
    private String getEmailContent(String orderNumber) {
        System.out.println("Retrieving email content for order: " + orderNumber);
        
        try {
            Properties props = new Properties();
            props.put("mail.store.protocol", "imap");
            props.put("mail.imap.host", MAIL_HOST);
            props.put("mail.imap.port", MAIL_PORT);
            props.put("mail.imap.ssl.enable", "true");
            
            Session session = Session.getDefaultInstance(props);
            Store store = session.getStore("imap");
            store.connect(MAIL_HOST, TEST_EMAIL, EMAIL_PASSWORD);
            
            Folder folder = store.getFolder("INBOX");
            folder.open(Folder.READ_ONLY);
            
            SearchTerm subjectTerm = new SubjectTerm("Order Confirmation");
            SearchTerm fromTerm = new FromTerm(new InternetAddress(EXPECTED_SENDER));
            SearchTerm combinedTerm = new AndTerm(subjectTerm, fromTerm);
            
            Message[] messages = folder.search(combinedTerm);
            
            String emailContent = "";
            for (Message msg : messages) {
                String content = getTextContent(msg);
                if (content.contains(orderNumber)) {
                    emailContent = content;
                    break;
                }
            }
            
            folder.close(false);
            store.close();
            
            return emailContent;
            
        } catch (Exception e) {
            System.err.println("Error retrieving email content: " + e.getMessage());
            return "";
        }
    }
    
    private void validateEmailContent(String emailContent, String orderNumber) {
        System.out.println("Validating email content...");
        
        // Basic content validation
        Assert.assertTrue(emailContent.contains("Marvel Electronics"), "Email should contain company name");
        Assert.assertTrue(emailContent.contains("Order Confirmation"), "Email should contain confirmation text");
        Assert.assertTrue(emailContent.contains(orderNumber), "Email should contain order number");
        Assert.assertTrue(emailContent.contains("Thank you"), "Email should contain thank you message");
        
        // Validate email structure
        Assert.assertTrue(emailContent.contains("Order Details"), "Email should contain order details section");
        Assert.assertTrue(emailContent.contains("Shipping Address"), "Email should contain shipping address");
        Assert.assertTrue(emailContent.contains("Total Amount"), "Email should contain total amount");
        
        // Validate contact information
        Assert.assertTrue(emailContent.contains("support@marvelelectronics.com") || 
                         emailContent.contains("customer service"), "Email should contain contact information");
        
        System.out.println("Email content validation completed successfully");
    }
    
    private int getEmailCount(String orderNumber) {
        System.out.println("Counting emails for order: " + orderNumber);
        
        try {
            Properties props = new Properties();
            props.put("mail.store.protocol", "imap");
            props.put("mail.imap.host", MAIL_HOST);
            props.put("mail.imap.port", MAIL_PORT);
            props.put("mail.imap.ssl.enable", "true");
            
            Session session = Session.getDefaultInstance(props);
            Store store = session.getStore("imap");
            store.connect(MAIL_HOST, TEST_EMAIL, EMAIL_PASSWORD);
            
            Folder folder = store.getFolder("INBOX");
            folder.open(Folder.READ_ONLY);
            
            SearchTerm subjectTerm = new SubjectTerm("Order Confirmation");
            SearchTerm fromTerm = new FromTerm(new InternetAddress(EXPECTED_SENDER));
            SearchTerm combinedTerm = new AndTerm(subjectTerm, fromTerm);
            
            Message[] messages = folder.search(combinedTerm);
            
            int count = 0;
            for (Message msg : messages) {
                String content = getTextContent(msg);
                if (content.contains(orderNumber)) {
                    count++;
                }
            }
            
            folder.close(false);
            store.close();
            
            System.out.println("Found " + count + " emails for order: " + orderNumber);
            return count;
            
        } catch (Exception e) {
            System.err.println("Error counting emails: " + e.getMessage());
            return 0;
        }
    }
    
    private String getTextContent(Message message) throws Exception {
        if (message.isMimeType("text/plain")) {
            return (String) message.getContent();
        } else if (message.isMimeType("text/html")) {
            return (String) message.getContent();
        } else if (message.isMimeType("multipart/*")) {
            Multipart multipart = (Multipart) message.getContent();
            return getTextFromMultipart(multipart);
        } else {
            return "";
        }
    }
    
    private String getTextFromMultipart(Multipart multipart) throws Exception {
        StringBuilder result = new StringBuilder();
        int count = multipart.getCount();
        for (int i = 0; i < count; i++) {
            BodyPart bodyPart = multipart.getBodyPart(i);
            if (bodyPart.isMimeType("text/plain")) {
                result.append(bodyPart.getContent().toString());
            } else if (bodyPart.isMimeType("text/html")) {
                result.append(bodyPart.getContent().toString());
            } else if (bodyPart.isMimeType("multipart/*")) {
                result.append(getTextFromMultipart((Multipart) bodyPart.getContent()));
            }
        }
        return result.toString();
    }
    
    @DataProvider(name = "orderTestData")
    public Object[][] getOrderTestData() {
        return new Object[][] {
            {"Electronics", "Smartphone Galaxy S24", "John Doe"},
            {"Electronics", "Laptop Pro 16", "Jane Smith"},
            {"Electronics", "Tablet Air 12", "Bob Johnson"},
            {"Home Entertainment", "Smart TV 55", "Alice Brown"},
            {"Home Entertainment", "Sound System Pro", "Charlie Wilson"}
        };
    }
    
    @AfterMethod
    public void cleanupAfterTest() {
        // Clear any test data or logout if needed
        try {
            if (driver.getCurrentUrl().contains("marvelelectronics.com")) {
                // Try to logout
                try {
                    WebElement logoutButton = driver.findElement(By.id("logout-btn"));
                    if (logoutButton.isDisplayed()) {
                        logoutButton.click();
                    }
                } catch (Exception e) {
                    // Logout button might not be present, continue
                }
            }
        } catch (Exception e) {
            System.out.println("Cleanup warning: " + e.getMessage());
        }
    }
    
    @AfterClass
    public void tearDown() {
        System.out.println("Tearing down test environment...");
        
        if (driver != null) {
            driver.quit();
            System.out.println("WebDriver closed successfully");
        }
    }
}
