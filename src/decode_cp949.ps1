param([string]$SourcePath,[string]$OutputPath)
$ErrorActionPreference='Stop'
$b=[IO.File]::ReadAllBytes($SourcePath)
$enc=[Text.Encoding]::GetEncoding(949,[Text.EncoderExceptionFallback]::new(),[Text.DecoderExceptionFallback]::new())
$s=$enc.GetString($b)
$back=$enc.GetBytes($s)
$sha=[Security.Cryptography.SHA256]::Create()
if([Convert]::ToBase64String($sha.ComputeHash($b)) -ne [Convert]::ToBase64String($sha.ComputeHash($back))){throw 'CP949 roundtrip mismatch'}
[IO.File]::WriteAllText($OutputPath,$s,[Text.UTF8Encoding]::new($false))
