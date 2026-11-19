# ========================================================
# MBCT Local API — PowerShell Command Examples
# ========================================================

Write-Host "`n--- 1. Chat Completion (Invoke-RestMethod) ---" -ForegroundColor Cyan
$body = @{
    model = "phi-3-mini"
    messages = @(
        @{ role = "user"; content = "Explain what a mutex is in one sentence." }
    )
    temperature = 0.5
    max_tokens = 50
    traits = @{
        directness = 0.95
        verbosity = 0.20
    }
} | ConvertTo-Json -Depth 5

$response = Invoke-RestMethod -Uri "http://127.0.0.1:8088/v1/chat/completions" -Method POST -ContentType "application/json" -Body $body
Write-Host "Response: $($response.choices[0].message.content)" -ForegroundColor Green
Write-Host "Steering Applied: $($response.mbct.steering_applied)" -ForegroundColor Yellow

Write-Host "`n--- 2. Simple Generate API ---" -ForegroundColor Cyan
$simpleBody = @{
    prompt = "What database do we use in production?"
    inject_memory = $true
    max_tokens = 60
} | ConvertTo-Json

$simpleResponse = Invoke-RestMethod -Uri "http://127.0.0.1:8088/api/generate" -Method POST -ContentType "application/json" -Body $simpleBody
Write-Host "Response: $($simpleResponse.response)" -ForegroundColor Green
Write-Host "Memory Applied: $($simpleResponse.memory_applied)" -ForegroundColor Yellow

Write-Host "`n--- 3. List Models ---" -ForegroundColor Cyan
$models = Invoke-RestMethod -Uri "http://127.0.0.1:8088/v1/models"
$models.data | Format-Table id, owned_by
