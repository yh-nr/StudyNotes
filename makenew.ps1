echo "Enter slide name: "
$slide = Read-Host


# if no slide name is entered, use 'NewSlide' as a default name
if ($slide -eq "") {
    $slide = "NewSlide"
}
# get current date/time in specified format
$dateTime = Get-Date -Format "yyMMddHHmmss"
# append datetime to slide name
$slide += "_" + $dateTime

New-Item -ItemType Directory -Force -Path "$PSScriptRoot\src\$slide"
Copy-Item -Path "$PSScriptRoot\src\template\index.md" -Destination "$PSScriptRoot\src\$slide\index.md" -Confirm:$false
