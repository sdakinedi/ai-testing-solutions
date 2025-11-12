from rag_mcp_server1 import generate_test_cases

# Email notification requirements based on earlier findings
requirements = """
The system requires:
1. Maintaining customer email information as part of the customer profile
2. Sending an order confirmation to the user via email
3. Email notifications for various system events
"""

print("Generating test cases for Email Notifications...")
print("=" * 60)

test_cases = generate_test_cases(
    feature="Email Notifications",
    requirements=requirements,
    context="E-commerce system with customer profiles and order management"
)

print(test_cases)
