$ErrorActionPreference = "Stop"

$outputPath = Join-Path $PSScriptRoot "..\data\urban_heat_islands.pdf"

$paragraphs = @(
    "Urban Heat Islands and Climate Adaptation",
    "An urban heat island occurs when a city becomes noticeably warmer than nearby rural areas. Buildings, roads, parking lots, and rooftops absorb solar energy during the day and release it slowly at night. Dense development can also reduce airflow, while vehicles, air conditioners, and industrial equipment add waste heat. The result is a city climate that can remain hot even after sunset. This matters because high nighttime temperatures prevent people from recovering from daytime heat stress.",
    "Heat islands are not evenly distributed. Neighborhoods with fewer trees, more pavement, older housing, and limited access to parks often experience higher temperatures. These areas may also have more residents who are elderly, medically vulnerable, or unable to afford air conditioning. Because of this, heat adaptation is both an environmental challenge and a public health challenge. A citywide average temperature can hide dangerous local hot spots.",
    "Trees are one of the most effective cooling strategies. They provide shade and cool the air through evapotranspiration, the process by which water moves from soil through plants and into the atmosphere. Street trees can make sidewalks more comfortable and reduce indoor cooling demand when they shade windows and walls. However, tree programs require maintenance, watering, species selection, and community input. Planting a tree is only the beginning; keeping it alive through drought and construction pressure is the harder task.",
    "Cool roofs and reflective pavements can also reduce heat absorption. A dark roof may become extremely hot in direct sun, while a cool roof reflects more sunlight and emits heat more effectively. This can lower indoor temperatures and reduce air conditioning use. Reflective pavement must be chosen carefully because glare and reflected heat can affect pedestrians. Permeable pavement can help by allowing water to infiltrate, although it must be maintained so pores do not clog.",
    "Heat adaptation also depends on emergency planning. Cities can open cooling centers, extend pool and library hours, send alerts in multiple languages, and check on vulnerable residents during heat waves. Transit stops need shade because people waiting for buses may be exposed for long periods. Outdoor workers need rest, water, and schedule adjustments. Schools and sports leagues need heat policies that protect children during extreme conditions.",
    "Good heat planning uses data. Satellite temperature maps, tree canopy surveys, hospital records, and resident reports can identify priority areas. The strongest plans combine physical changes, such as shade and cool surfaces, with social programs that reach people before heat becomes deadly. Reducing an urban heat island is not a single project. It is a long-term practice of redesigning streets, buildings, emergency systems, and maintenance budgets around a hotter climate.",
    "Public communication is part of adaptation because heat risk is often underestimated. People may prepare for storms more readily than for several extremely hot days, even though heat can be deadly. Cities can publish neighborhood heat maps, name cooling centers clearly, and work with local organizations that residents already trust. When communities understand where heat risk is highest and what help is available, infrastructure investments and emergency response become more effective."
)

function Escape-PdfText([string]$text) {
    return $text.Replace("\", "\\").Replace("(", "\(").Replace(")", "\)")
}

$objects = New-Object System.Collections.Generic.List[string]
$objects.Add("<< /Type /Catalog /Pages 2 0 R >>")
$objects.Add("<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>")
$objects.Add("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>")
$objects.Add("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>")
$objects.Add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

$page1 = $paragraphs[0..3]
$page2 = $paragraphs[4..7]

function Build-PageStream($lines) {
    $content = "BT`n/F1 12 Tf`n50 742 Td`n14 TL`n"
    foreach ($paragraph in $lines) {
        $words = $paragraph.Split(" ")
        $line = ""
        foreach ($word in $words) {
            $candidate = if ($line) { "$line $word" } else { $word }
            if ($candidate.Length -gt 82) {
                $content += "(" + (Escape-PdfText $line) + ") Tj`nT*`n"
                $line = $word
            } else {
                $line = $candidate
            }
        }
        if ($line) {
            $content += "(" + (Escape-PdfText $line) + ") Tj`nT*`n"
        }
        $content += "T*`n"
    }
    $content += "ET"
    return $content
}

$stream1 = Build-PageStream $page1
$stream2 = Build-PageStream $page2
$objects.Add("<< /Length $($stream1.Length) >>`nstream`n$stream1`nendstream")
$objects.Add("<< /Length $($stream2.Length) >>`nstream`n$stream2`nendstream")

$bytes = New-Object System.Collections.Generic.List[byte]
function Add-Ascii([string]$text) {
    $bytes.AddRange([System.Text.Encoding]::ASCII.GetBytes($text))
}

Add-Ascii "%PDF-1.4`n"
$offsets = New-Object System.Collections.Generic.List[int]
for ($i = 0; $i -lt $objects.Count; $i++) {
    $offsets.Add($bytes.Count)
    Add-Ascii "$($i + 1) 0 obj`n$($objects[$i])`nendobj`n"
}
$xrefOffset = $bytes.Count
Add-Ascii "xref`n0 $($objects.Count + 1)`n"
Add-Ascii "0000000000 65535 f `n"
foreach ($offset in $offsets) {
    Add-Ascii ("{0:D10} 00000 n `n" -f $offset)
}
Add-Ascii "trailer`n<< /Size $($objects.Count + 1) /Root 1 0 R >>`nstartxref`n$xrefOffset`n%%EOF"

[System.IO.File]::WriteAllBytes($outputPath, $bytes.ToArray())
Write-Host "Created $outputPath"
