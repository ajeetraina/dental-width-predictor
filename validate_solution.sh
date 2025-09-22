#!/bin/bash
# docker_test_working_e_space.sh
# Test the working E-space system using Docker Compose

set -e

echo "🦷 TESTING WORKING E-SPACE SYSTEM WITH DOCKER COMPOSE"
echo "====================================================="

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check if container is running
if ! docker ps | grep -q "dental-width-predictor"; then
    echo "❌ Docker container not running. Starting it..."
    docker-compose up -d dental-width-predictor
    sleep 5
fi

echo "✅ Docker services ready"

# Test 1: Check if the working analyzer exists
echo ""
echo "📝 TEST 1: Check working E-space analyzer exists"
echo "------------------------------------------------"
docker exec dental-width-predictor ls -la /app/src/working_e_space_analyzer.py || {
    echo "❌ Working analyzer not found. Creating it..."
    # Copy from local if it exists
    if [ -f "src/working_e_space_analyzer.py" ]; then
        docker cp src/working_e_space_analyzer.py dental-width-predictor:/app/src/
        echo "✅ Copied working analyzer to container"
    else
        echo "❌ Working analyzer not found locally either"
        exit 1
    fi
}

# Test 2: Test on AVANISHK sample (known working case)
echo ""
echo "🧪 TEST 2: Test on AVANISHK sample (validated case)"
echo "------------------------------------------------"
docker exec dental-width-predictor python /app/src/working_e_space_analyzer.py \
    --input "/app/data/samples/AVANISHK 11 YRS MALE_DR SAHEEB JAN_2013_07_22_2D_Image_Shot.jpg" \
    --output "/app/test_results" \
    --calibration 0.1 \
    --debug

# Test 3: Check results were generated
echo ""
echo "📊 TEST 3: Check results were generated"
echo "---------------------------------------"
docker exec dental-width-predictor ls -la /app/test_results/

# Test 4: Display JSON results
echo ""
echo "📄 TEST 4: Display JSON results"
echo "-------------------------------"
docker exec dental-width-predictor find /app/test_results -name "*.json" -exec cat {} \;

# Test 5: Test batch processing on first 3 samples
echo ""
echo "🔄 TEST 5: Test batch processing (first 3 samples)"
echo "-------------------------------------------------"
docker exec dental-width-predictor bash -c "
# Create a temporary directory with just first 3 samples
mkdir -p /app/test_batch_3
cd /app/data/samples
ls *.jpg | head -3 | while read file; do
    cp \"\$file\" /app/test_batch_3/
done

# Run batch processing
python /app/src/working_e_space_analyzer.py \
    --input /app/test_batch_3 \
    --output /app/batch_test_results \
    --calibration 0.1 \
    --debug
"

# Test 6: Show batch results summary
echo ""
echo "📈 TEST 6: Batch results summary"
echo "--------------------------------"
docker exec dental-width-predictor bash -c "
echo 'JSON files created:'
ls -la /app/batch_test_results/*.json 2>/dev/null || echo 'No JSON files found'

echo ''
echo 'CSV summary:'
if [ -f /app/batch_test_results/working_e_space_batch_summary.csv ]; then
    head -10 /app/batch_test_results/working_e_space_batch_summary.csv
else
    echo 'No CSV summary found'
fi
"

# Test 7: Copy results to host for inspection
echo ""
echo "💾 TEST 7: Copy results to host"
echo "-------------------------------"
docker cp dental-width-predictor:/app/test_results ./docker_test_results
docker cp dental-width-predictor:/app/batch_test_results ./docker_batch_test_results

echo "✅ Results copied to:"
echo "   - ./docker_test_results (single image test)"
echo "   - ./docker_batch_test_results (batch test)"

# Test 8: Performance test - measure processing time
echo ""
echo "⚡ TEST 8: Performance test"
echo "--------------------------"
docker exec dental-width-predictor bash -c "
echo 'Testing processing speed...'
time python /app/src/working_e_space_analyzer.py \
    --input '/app/data/samples/AVANISHK 11 YRS MALE_DR SAHEEB JAN_2013_07_22_2D_Image_Shot.jpg' \
    --output /app/perf_test \
    --calibration 0.1 > /dev/null 2>&1
"

# Test 9: Validation against expected AVANISHK results
echo ""
echo "✅ TEST 9: Validate AVANISHK results"
echo "------------------------------------"
if [ -f "./docker_test_results/AVANISHK_11_YRS_MALE_DR_SAHEEB_JAN_2013_07_22_2D_Image_Shot_working_e_space.json" ]; then
    echo "Found AVANISHK results JSON. Checking values..."
    python3 -c "
import json
import sys

try:
    with open('./docker_test_results/AVANISHK_11_YRS_MALE_DR_SAHEEB_JAN_2013_07_22_2D_Image_Shot_working_e_space.json', 'r') as f:
        data = json.load(f)
    
    quadrants = data.get('quadrants', {})
    summary = data.get('summary', {})
    
    print(f'📊 AVANISHK Test Results:')
    print(f'   Successful quadrants: {summary.get(\"successful_quadrants\", 0)}/4')
    print(f'   Average E-space: {summary.get(\"average_e_space_mm\", 0):.2f}mm')
    print(f'   Success rate: {summary.get(\"success_rate_percent\", 0):.1f}%')
    
    print(f'')
    print(f'📏 Individual quadrant results:')
    for quadrant, values in quadrants.items():
        e_space = values.get('e_space_mm', 0)
        primary_w = values.get('primary_molar', {}).get('width_mm', 0)
        premolar_w = values.get('premolar', {}).get('width_mm', 0)
        print(f'   {quadrant:12}: Primary={primary_w:5.2f}mm, Premolar={premolar_w:5.2f}mm → E-space={e_space:5.2f}mm')
    
    # Validation checks
    total_measurements = len(quadrants)
    avg_e_space = summary.get('average_e_space_mm', 0)
    
    print(f'')
    print(f'✅ VALIDATION RESULTS:')
    
    if total_measurements >= 2:
        print(f'   ✅ Multiple measurements generated ({total_measurements})')
    else:
        print(f'   ⚠️  Only {total_measurements} measurements (expected 2-4)')
    
    if 1.0 <= avg_e_space <= 6.0:
        print(f'   ✅ Average E-space in clinical range ({avg_e_space:.2f}mm)')
    else:
        print(f'   ❌ Average E-space out of range ({avg_e_space:.2f}mm)')
    
    # Check individual values
    all_e_spaces = [v['e_space_mm'] for v in quadrants.values()]
    if all(0.3 <= e <= 5.0 for e in all_e_spaces):
        print(f'   ✅ All E-space values in clinical range (0.3-5.0mm)')
    else:
        print(f'   ❌ Some E-space values out of clinical range')
    
    print(f'')
    if total_measurements >= 2 and 1.0 <= avg_e_space <= 6.0:
        print(f'🎉 OVERALL: WORKING E-SPACE SYSTEM IS FUNCTIONING CORRECTLY!')
        sys.exit(0)
    else:
        print(f'⚠️  OVERALL: Results need review')
        sys.exit(1)

except Exception as e:
    print(f'❌ Error reading results: {e}')
    sys.exit(1)
"
else
    echo "❌ AVANISHK results JSON not found"
fi

# Final summary
echo ""
echo "📋 FINAL TEST SUMMARY"
echo "===================="
echo "✅ Docker Compose E-space testing completed"
echo ""
echo "📁 Generated files:"
echo "   - ./docker_test_results/ (single image results)"
echo "   - ./docker_batch_test_results/ (batch processing results)"
echo ""
echo "🔍 To inspect results:"
echo "   cat ./docker_test_results/*.json"
echo "   cat ./docker_batch_test_results/*.csv"
echo ""
echo "🚀 Next steps:"
echo "   1. Review the JSON results above"
echo "   2. Check debug visualizations if generated"
echo "   3. Run on additional samples if needed"
echo "   4. Deploy to production if validation successful"

echo ""
echo "🎯 Working E-space system Docker testing complete!"
