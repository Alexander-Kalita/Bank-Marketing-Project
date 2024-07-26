
import requests
import unittest

# Conduct integration test
class TestIntegration(unittest.TestCase):
    def test_model_integration(self):
        features = {
            'age': 18,
            'balance': 300,
            'day': 5,
            'duration': 500,
            'campaign': 1,
            'pdays': -1,
            'previous': 4,
            'job_blue-collar': 0,
            'job_entrepreneur': 0,
            'job_housemaid': 0,
            'job_management': 1,
            'job_retired': 0,
            'job_self-employed': 0,
            'job_services': 0,
            'job_student': 0,
            'job_technician': 0,
            'job_unemployed': 0,
            'job_unknown': 0,
            'education_secondary': 1,
            'education_tertiary': 0,
            'education_unknown': 0,
            'marital_married': 1,
            'marital_single': 0,
            'default_yes': 0,
            'housing_yes': 1,
            'loan_yes': 0,
            'contact_telephone': 1,
            'contact_unknown': 0,
            'month_aug': 0,
            'month_dec': 0,
            'month_feb': 0,
            'month_jan': 0,
            'month_jul': 1,
            'month_jun': 0,
            'month_mar': 0,
            'month_may': 0,
            'month_nov': 0,
            'month_oct': 0,
            'month_sep': 0,
            'poutcome_other': 0,
            'poutcome_success': 0,
            'poutcome_unknown': 1
        }
        endpoint = "http://localhost:8081/predict"
        response = requests.post(endpoint, json={'features': features})
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        prediction = response_json.get('prediction', None)
        self.assertIsNotNone(prediction)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)
unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))
