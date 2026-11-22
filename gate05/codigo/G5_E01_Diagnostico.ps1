$ErrorActionPreference = 'Continue'
$cpu = Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
if (-not $cpu) {
    $cpu = [ordered]@{
        Name = Get-ItemPropertyValue 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0' ProcessorNameString
        NumberOfLogicalProcessors = (Get-ChildItem 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor' | Measure-Object).Count
    }
}
$os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue | Select-Object Caption, Version, OSArchitecture, TotalVisibleMemorySize, FreePhysicalMemory
if (-not $os) {
    $version = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
    $os = [ordered]@{ DisplayVersion=$version.DisplayVersion; Build="$($version.CurrentBuild).$($version.UBR)"; Architecture=$env:PROCESSOR_ARCHITECTURE }
}
$gpu = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Select-Object Name, DriverVersion
if (-not $gpu) {
    $gpu = Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}' |
        Where-Object { $_.PSChildName -match '^\d{4}$' } |
        Get-ItemProperty | Where-Object DriverDesc | Select-Object DriverDesc, DriverVersion
}
$disk = Get-Item -LiteralPath $PSScriptRoot | ForEach-Object { Get-PSDrive -Name $_.PSDrive.Name } | Select-Object Used, Free
$python = @()
foreach ($name in @('python','python3','py')) {
    $command = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $command) { continue }
    if ($command.Source -like '*\Microsoft\WindowsApps\*' -and $name -ne 'py') {
        $python += [ordered]@{ comando=$name; estado='Alias do Windows detectado, nao executado' }
        continue
    }
    $probe = @'
import ctypes, importlib.util, json, platform
r = dict(python=platform.python_version(), architecture=platform.architecture()[0])
memory = (ctypes.c_uint64 * 8)()
memory[0] = ctypes.sizeof(memory)
if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
    r.update(ram_total_bytes=memory[1], ram_available_bytes=memory[2])
for p in ('numpy', 'torch', 'torchvision'):
    r[p] = importlib.util.find_spec(p) is not None
if r['torch']:
    try:
        import torch
        r.update(torch_version=torch.__version__, cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(), cuda_devices=torch.cuda.device_count())
    except Exception as e:
        r['torch_check_error'] = type(e).__name__
print(json.dumps(r))
'@
    $result = & $command.Source -I -c $probe 2>$null
    if ($LASTEXITCODE -eq 0) { $python += [ordered]@{ comando=$name; ambiente=($result | ConvertFrom-Json) } }
    else { $python += [ordered]@{ comando=$name; estado='Interpretador indisponivel'; codigo=$LASTEXITCODE } }
}
$nvidia = Get-Command nvidia-smi -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
$nvidiaInfo = if ($nvidia) { & $nvidia.Source '--query-gpu=name,driver_version,memory.total' '--format=csv,noheader' 2>$null }
[ordered]@{ cpu=@($cpu); windows=$os; gpu=@($gpu); disco_bytes=$disk; python=$python; nvidia=$nvidiaInfo } | ConvertTo-Json -Depth 6
