' MT5_to_Excel.vbs - One-click Excel analysis
Set objExcel = CreateObject("Excel.Application")
objExcel.Visible = True

' Open CSV from MT5
mt5Path = "C:\Users\davidug.SIFAXGROUP\AppData\Roaming\MetaQuotes\Terminal\BA78EA1631820D7AEF052166A20E7B1A\MQL5\Files\StateLog.csv"

If Not CreateObject("Scripting.FileSystemObject").FileExists(mt5Path) Then
    MsgBox "MT5 file not found:" & vbCrLf & mt5Path
    WScript.Quit
End If

Set objWorkbook = objExcel.Workbooks.Open(mt5Path)
Set objSheet = objWorkbook.Worksheets(1)

' Add Cluster column
objSheet.Cells(1, 14).Value = "Cluster"
objSheet.Range("N2").Formula = "=D2&""|""&E2&""|""&F2&""|""&G2"
objSheet.Range("N2").AutoFill objSheet.Range("N2:N" & objSheet.UsedRange.Rows.Count)

' Add analysis
objSheet.Cells(1, 15).Value = "Persistence_Score"
objSheet.Cells(1, 16).Value = "Duration_Hours"

MsgBox "Analysis ready! Check column N for clusters."
