"""
Test the enhanced API with caching and monitoring
"""
import requests
import time

BASE_URL = "http://localhost:8000"

print("="*60)
print("Testing Enhanced Sentiment Analysis API")
print("="*60)

# Test 1: Health Check
print("\n1. Health Check")
response = requests.get(f"{BASE_URL}/health")
print(f"   Status: {response.json()}")

# Test 2: Model Info (with cache stats)
print("\n2. Model Info + Cache Stats")
response = requests.get(f"{BASE_URL}/model/info")
info = response.json()
print(f"   Model: {info['model_type']}")
print(f"   Device: {info['device']}")
print(f"   Cache: {info['cache']}")

# Test 3: First Prediction (Cache MISS)
print("\n3. First Prediction (Cache MISS)")
text1 = "This is an amazing product! I absolutely love it!"
start = time.time()
response = requests.post(
    f"{BASE_URL}/predict",
    json={"text": text1}
)
duration1 = time.time() - start
result1 = response.json()
print(f"   Text: {text1}")
print(f"   Sentiment: {result1['sentiment']} ({result1['confidence']:.2%})")
print(f"   Duration: {duration1*1000:.2f}ms")

# Test 4: Same Prediction (Cache HIT - should be faster!)
print("\n4. Same Prediction (Cache HIT)")
start = time.time()
response = requests.post(
    f"{BASE_URL}/predict",
    json={"text": text1}
)
duration2 = time.time() - start
result2 = response.json()
print(f"   Sentiment: {result2['sentiment']} ({result2['confidence']:.2%})")
print(f"   Duration: {duration2*1000:.2f}ms")
print(f"   Speedup: {duration1/duration2:.2f}x faster!")

# Test 5: Different Prediction
print("\n5. New Prediction (Cache MISS)")
text2 = "This is terrible. Worst experience ever."
response = requests.post(
    f"{BASE_URL}/predict",
    json={"text": text2}
)
result3 = response.json()
print(f"   Text: {text2}")
print(f"   Sentiment: {result3['sentiment']} ({result3['confidence']:.2%})")

# Test 6: Prometheus Metrics
print("\n6. Prometheus Metrics")
response = requests.get(f"{BASE_URL}/metrics")
metrics = response.text
print(f"   Total lines: {len(metrics.splitlines())}")
print(f"   Sample metrics:")
for line in metrics.splitlines()[:10]:
    if not line.startswith('#'):
        print(f"     {line}")

print("\n" + "="*60)
print("✓ All tests completed!")
print("="*60)
print("\nOpen these dashboards:")
print("  • Prometheus: http://localhost:9090")
print("  • Grafana: http://localhost:3000 (admin/admin)")
print("  • API Docs: http://localhost:8000/docs")
