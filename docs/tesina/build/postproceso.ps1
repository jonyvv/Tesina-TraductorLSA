# =============================================================================
# docs/tesina/build/postproceso.ps1
# -----------------------------------------------------------------------------
# Abre el .docx que genera build.js con Microsoft Word, actualiza los campos
# (índice general, de figuras, de tablas, numeración de leyendas y páginas),
# guarda y exporta el PDF. Imprime la cantidad de páginas, que es el número que
# hay que mirar contra el rango 90-120 pedido.
#
# Uso (desde docs/tesina/build):
#     powershell -ExecutionPolicy Bypass -File postproceso.ps1
#
# Requiere Word instalado (usa automatización COM).
# =============================================================================
$ErrorActionPreference = "Stop"
$salida = Join-Path $PSScriptRoot "..\salida"
$docx = (Resolve-Path (Join-Path $salida "Tesina-TraductorLSA.docx")).Path
$pdf = [System.IO.Path]::ChangeExtension($docx, ".pdf")

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx)
    # Dos pasadas: la primera numera leyendas y arma los índices; la segunda
    # corrige los números de página que se corren al insertar los índices.
    for ($i = 0; $i -lt 2; $i++) {
        $doc.Fields.Update() | Out-Null
        foreach ($toc in $doc.TablesOfContents) { $toc.Update() | Out-Null }
        foreach ($tof in $doc.TablesOfFigures) { $tof.Update() | Out-Null }
    }
    $doc.Save()
    # 17 = wdExportFormatPDF
    $doc.ExportAsFixedFormat($pdf, 17)
    # 2 = wdStatisticPages, 0 = wdStatisticWords
    $paginas = $doc.ComputeStatistics(2)
    $palabras = $doc.ComputeStatistics(0)
    Write-Output "Paginas: $paginas"
    Write-Output "Palabras: $palabras"
    Write-Output "PDF: $pdf"
    # Copia con el nombre que pide la facultad para subir al Drive:
    # apellidos + fecha + version, p. ej. ProvenzanoVidela20261001v01.pdf
    $version = if ($env:VERSION) { $env:VERSION } else { "01" }
    $entrega = Join-Path (Split-Path $pdf) ("ProvenzanoVidela{0}v{1}.pdf" -f (Get-Date -Format "yyyyMMdd"), $version)
    Copy-Item $pdf $entrega -Force
    Write-Output "Copia para entregar: $entrega"
    $doc.Close()
}
finally {
    $word.Quit()
}
