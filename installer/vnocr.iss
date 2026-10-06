; Inno Setup script for VNOCR (Windows 11 on ARM64). Built by build_installer.ps1.
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{6B0E2C1D-7F2B-4C55-9A0E-5D1A3F4B8C21}
AppName=VNOCR - Vietnamese Document OCR
AppVersion={#AppVersion}
AppPublisher=VNOCR
DefaultDirName={autopf}\VNOCR
DefaultGroupName=VNOCR
; Native ARM64 only: the NPU runtime (QNN) has no x64 build.
ArchitecturesAllowed=arm64
ArchitecturesInstallIn64BitMode=arm64
; Install for the current user without admin rights unless the user chooses "all users".
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=VNOCR-Setup-{#AppVersion}-arm64
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=VNOCR - Vietnamese Document OCR
DisableProgramGroupPage=yes

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\VNOCR\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\VNOCR"; Filename: "{app}\VNOCR.exe"
Name: "{group}\VNOCR self-test (NPU check)"; Filename: "{app}\VNOCR.exe"; Parameters: "--selftest"
Name: "{group}\Uninstall VNOCR"; Filename: "{uninstallexe}"
Name: "{autodesktop}\VNOCR"; Filename: "{app}\VNOCR.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\VNOCR.exe"; Description: "Start VNOCR"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Compiled NPU contexts and the log file.
Type: filesandordirs; Name: "{localappdata}\vnocr"
