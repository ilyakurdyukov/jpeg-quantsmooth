/*
 * Copyright (C) 2020 Kurdyukov Ilya
 *
 * This file is part of jpeg quantsmooth (windows GUI wrapper)
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "dialog.h"

#define UNICODE
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commdlg.h>

#ifdef WITH_DROP
#include <shellapi.h>
#endif

#pragma GCC diagnostic ignored "-Wformat"

#define FNLEN_MAX 1024

static wchar_t ofnbuf[FNLEN_MAX], sfnbuf[FNLEN_MAX];
static OPENFILENAME ofn, sfn;
static const wchar_t *appname = L"JPEGQS Wrapper";
static LPCWSTR jpegqs_exe;
static HWND hwndDlg = NULL;
static HANDLE hErrRead = INVALID_HANDLE_VALUE;
static HANDLE infoThread = INVALID_HANDLE_VALUE;

static CRITICAL_SECTION CriticalSection[1];
#define LOCK(i) EnterCriticalSection(&CriticalSection[i]);
#define UNLOCK(i) LeaveCriticalSection(&CriticalSection[i]);

#define OPTLEN 128
#define OUTINCR (1 << 18)
#define LOGINCR 16

static char *outmem = NULL;
static size_t outcur, outmax;
static char *logmem = NULL;
static size_t logcur, logmax;
static char options[OPTLEN];

static void log_grow(size_t n) {
	if (logcur + n >= logmax) {
		logmax = (logcur + n + LOGINCR) & -LOGINCR;
		logmem = realloc(logmem, logmax);
		if (!logmem) return;
	}
}

static void log_update(size_t n) {
	if (hwndDlg) {
		// SetDlgItemTextA(hwndDlg, IDC_CONSOLE, logmem);
		// update and scroll to bottom
		(void)n;
		HWND edit = GetDlgItem(hwndDlg, IDC_CONSOLE);
		SendMessageA(edit, EM_SETSEL, 0, -1);
		SendMessageA(edit, EM_REPLACESEL, 0, (LPARAM)logmem);
	}
}

static void log_append(const char *str, int n) {
	// LOCK(1)
	if (!logmem) return;
	if (n < 0) n = strlen(str);
	log_grow(n);
	memcpy(logmem + logcur, str, n);
	logcur += n; logmem[logcur] = 0;
	log_update(n);
	// UNLOCK(1)
}

static DWORD WINAPI consoleThread(LPVOID lpParam) {
	int ret;
	(void)lpParam;
	for (;;) {
		DWORD rwCnt = 0;
		ret = PeekNamedPipe(hErrRead, NULL, 0, NULL, &rwCnt, NULL);
		if (!ret) break;
		if (!rwCnt) {
			Sleep(100);
			continue;
		}
		log_grow(rwCnt);
		ret = ReadFile(hErrRead, logmem + logcur, rwCnt, &rwCnt, NULL);
		if (!ret || !rwCnt) break;
		logcur += rwCnt; logmem[logcur] = 0;
		log_update(rwCnt);
	}
	CloseHandle(hErrRead);
	{
		char strbuf[80];
		snprintf(strbuf, sizeof(strbuf), "Output size: %i\r\n", (int)outcur);
		log_append(strbuf, -1);
	}
	return 0;
}

static void closeHandles() {
	if (outmem) {
		free(outmem);
		outmem = NULL;
	}
	if (infoThread != INVALID_HANDLE_VALUE) {
		WaitForSingleObject(infoThread, INFINITE);
		CloseHandle(infoThread);
	}
}

static int findfn(const wchar_t *path, int n) {
	if (n < 0) n = wcslen(path);
	while (n > 0) {
		wchar_t a = path[n - 1];
		if (a == '\\' || a == '/') break;
		n--;
	}
	return n;
}

static void cbSave() {
	int ret;
	LOCK(0)
	if (outmem && outcur) {
		if (!sfnbuf[0]) wcscpy(sfnbuf, ofnbuf);
		if (hwndDlg) {
			int n = findfn(sfnbuf, -1);
			GetDlgItemText(hwndDlg, IDC_FILENAME, sfnbuf + n, FNLEN_MAX - n);
		}
		ret = GetSaveFileName(&sfn);
		if (ret) {
			DWORD rwCnt;
			HANDLE outHandle = CreateFile(sfnbuf, GENERIC_WRITE,
					0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
			if (outHandle != INVALID_HANDLE_VALUE) {
				ret = WriteFile(outHandle, outmem, outcur, &rwCnt, NULL);
				CloseHandle(outHandle);
			}
		}
	}
	UNLOCK(0)
}

static void cbLoad(int use_ofn) {
	int ret = 1;
	LOCK(0)
	if (use_ofn) ret = GetOpenFileName(&ofn);
	if (ret) {
		SECURITY_ATTRIBUTES saAttr;
		STARTUPINFO si; PROCESS_INFORMATION pi;
		HANDLE hOutRead = INVALID_HANDLE_VALUE;
		HANDLE hErrWrite = INVALID_HANDLE_VALUE, hOutWrite = hErrRead;
		DWORD threadId;

		closeHandles();
		if (hwndDlg) {
			int n = findfn(ofnbuf, -1);
			SetDlgItemText(hwndDlg, IDC_FILENAME, ofnbuf + n);
			GetDlgItemTextA(hwndDlg, IDC_OPTIONS, options, OPTLEN);
			EnableWindow(GetDlgItem(hwndDlg, IDC_FILENAME), TRUE);
			EnableWindow(GetDlgItem(hwndDlg, IDC_LOAD), FALSE);
			EnableWindow(GetDlgItem(hwndDlg, IDC_SAVE), FALSE);
			SetDlgItemText(hwndDlg, IDC_STATUS, L"Processing...");
		}

		saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
		saAttr.bInheritHandle = TRUE;
		saAttr.lpSecurityDescriptor = NULL;

		if (logmem) {
			CreatePipe(&hErrRead, &hErrWrite, &saAttr, 0);
			SetHandleInformation(hErrRead, HANDLE_FLAG_INHERIT, 0);
		}
		CreatePipe(&hOutRead, &hOutWrite, &saAttr, 0);
		SetHandleInformation(hOutRead, HANDLE_FLAG_INHERIT, 0);

		if (logmem) {
			size_t inp_size = 0; char strbuf[80];
			snprintf(strbuf, sizeof(strbuf), "Loading \"%S\"\r\n", ofnbuf + findfn(ofnbuf, -1));
			log_append(strbuf, -1);
			HANDLE hFile = CreateFile(ofnbuf, GENERIC_READ,
					FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
			if (hFile != INVALID_HANDLE_VALUE) {
				inp_size = GetFileSize(hFile, NULL);
				CloseHandle(hFile);
			}
			snprintf(strbuf, sizeof(strbuf), "Input size: %i\r\n", (int)inp_size);
			log_append(strbuf, -1);
			log_append("Processing...\r\n", -1);
		}

		memset(&si, 0, sizeof(si));
		memset(&pi, 0, sizeof(pi));
		si.cb = sizeof(si);
		si.dwFlags = STARTF_USESTDHANDLES;
		si.hStdInput = INVALID_HANDLE_VALUE;
		si.hStdOutput = hOutWrite;
		si.hStdError = hErrWrite;
		{
			wchar_t jpegqs_cmd[FNLEN_MAX + OPTLEN + 40];
			snwprintf(jpegqs_cmd, sizeof(jpegqs_cmd) / sizeof(jpegqs_cmd[0]),
					L"%s %S -- \"%s\" -", jpegqs_exe, options, ofnbuf);
			ret = CreateProcess(NULL, jpegqs_cmd, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
		}
		CloseHandle(pi.hProcess);
		CloseHandle(pi.hThread);
		CloseHandle(hErrWrite);
		CloseHandle(hOutWrite);

		if (!ret) {
			wchar_t strbuf[80];
			snwprintf(strbuf, sizeof(strbuf) / sizeof(strbuf[0]),
				L"CreateProcess(\"%s\") failed with code %i\n", jpegqs_exe, GetLastError());
			MessageBox(hwndDlg, strbuf, appname, MB_OK);
			CloseHandle(hErrRead);
		} else {
			if (hErrRead != INVALID_HANDLE_VALUE)
				infoThread = CreateThread(NULL, 0, consoleThread, NULL, 0, &threadId);

			outmem = malloc(OUTINCR);
			outmax = OUTINCR; outcur = 0;
			for (;;) {
				DWORD rwCnt;
				ret = ReadFile(hOutRead, outmem + outcur, outmax - outcur, &rwCnt, NULL);
				if (!ret || !rwCnt) break;
				outcur += rwCnt;
				if (outcur == outmax) {
					outmem = realloc(outmem, outmax + OUTINCR);
					outmax += OUTINCR;
				}
			}
		}

		SetDlgItemText(hwndDlg, IDC_STATUS, L"");
		CloseHandle(hOutRead);

		if (hwndDlg) {
			EnableWindow(GetDlgItem(hwndDlg, IDC_LOAD), TRUE);
			if (outcur)
				EnableWindow(GetDlgItem(hwndDlg, IDC_SAVE), TRUE);
		}
	}
	UNLOCK(0)
}

INT_PTR WINAPI DialogProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
	(void)lParam;

	switch (msg) {
		case WM_INITDIALOG:
				hwndDlg = hwnd;
				ofn.hwndOwner = sfn.hwndOwner = hwnd;
				SetDlgItemTextA(hwnd, IDC_OPTIONS, options);
				SetDlgItemText(hwnd, IDC_FILENAME, L"");
				EnableWindow(GetDlgItem(hwnd, IDC_LOAD), TRUE);
				EnableWindow(GetDlgItem(hwnd, IDC_SAVE), FALSE);
				EnableWindow(GetDlgItem(hwnd, IDC_FILENAME), FALSE);
#ifdef WITH_DROP
				DragAcceptFiles(hwnd, TRUE);
#endif
				return TRUE;

		case WM_COMMAND:
			if (HIWORD(wParam) == BN_CLICKED) {
				switch (LOWORD(wParam)) {
					case IDC_LOAD: cbLoad(1); break;
					case IDC_SAVE: cbSave(); break;
				}
			}
			break;

#ifdef WITH_DROP
		case WM_DROPFILES: {
			HDROP hDrop = (HDROP)wParam;
			int n = DragQueryFile(hDrop, -1, NULL, 0);
			if (n != 1) {
				MessageBox(hwnd, L"Multiple file drop unsupported.", appname, MB_OK);
			} else {
				if (DragQueryFile(hDrop, 0, ofnbuf, FNLEN_MAX)) cbLoad(0);
			}
			DragFinish(hDrop);
			break;
		}
#endif

		case WM_DESTROY:
			break;

		case WM_CLOSE:
			GetDlgItemTextA(hwnd, IDC_OPTIONS, options, OPTLEN);
			EndDialog(hwnd, 0);
			break;
	}
	return FALSE;
}

int WINAPI WinMain(HINSTANCE hInstance,
		HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
	(void)hInstance; (void)hPrevInstance; (void)lpCmdLine; (void)nCmdShow;

	const wchar_t *fnfilt = L"JPEG image (*.jpg, *.jpeg)\0*.jpg;*jpeg\0All files\0*.*\0";
	const char *regkey = "Software\\JPEG Quant Smooth";

	{
		int n = GetModuleFileName(NULL, ofnbuf, FNLEN_MAX);
		n = findfn(ofnbuf, n);
		if (n) {
			ofnbuf[n] = 0;
			SetCurrentDirectory(ofnbuf);
		}
	}

	memset(&ofn, 0, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.lpstrFilter = fnfilt;
	ofn.nMaxFile = FNLEN_MAX;
	memcpy(&sfn, &ofn, sizeof(ofn));

	ofn.lpstrFile = ofnbuf;
	sfn.lpstrFile = sfnbuf;
	ofnbuf[0] = sfnbuf[0] = 0;
	ofn.lpstrTitle = L"Open JPEG image";
	ofn.Flags = OFN_HIDEREADONLY | OFN_FILEMUSTEXIST | 
		OFN_PATHMUSTEXIST | OFN_EXPLORER | OFN_NOCHANGEDIR;
	sfn.lpstrTitle = L"Save JPEG image";
	sfn.Flags = OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT | 
		OFN_PATHMUSTEXIST | OFN_EXPLORER | OFN_NOCHANGEDIR;

#ifdef __x86_64__
	jpegqs_exe = L"jpegqs64.exe";
#else
	jpegqs_exe = L"jpegqs.exe";
#endif

	{
		DWORD size = OPTLEN;
		LSTATUS status = RegGetValueA(HKEY_CURRENT_USER,
				regkey, "options", RRF_RT_REG_SZ, NULL, options, &size);
		if (status != ERROR_SUCCESS)
			strcpy(options, "--optimize --info 8 --quality 3");
	}

	InitializeCriticalSection(&CriticalSection[0]);

	logmem = malloc(LOGINCR);
	logmax = LOGINCR; logcur = 0;
	DialogBoxParam(0, MAKEINTRESOURCE(IDD_DIALOG), NULL, (DLGPROC)DialogProc, (LPARAM)NULL);

	{
		HKEY hKey;
		LSTATUS status = RegCreateKeyExA(HKEY_CURRENT_USER, regkey, 0, NULL,
				REG_OPTION_NON_VOLATILE, KEY_WRITE | KEY_QUERY_VALUE, NULL, &hKey, NULL);
		if (status == ERROR_SUCCESS) {
			RegSetValueExA(hKey, "options", 0, REG_SZ, (LPBYTE)options, strlen(options) + 1);
			RegCloseKey(hKey);
		}
	}

	LOCK(0)
	closeHandles();
	UNLOCK(0)
	if (logmem) free(logmem);

	DeleteCriticalSection(&CriticalSection[0]);
	return 0;
}
