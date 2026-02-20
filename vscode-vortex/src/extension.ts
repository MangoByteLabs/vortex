import * as vscode from 'vscode';
import { exec } from 'child_process';

export function activate(context: vscode.ExtensionContext) {
    const outputChannel = vscode.window.createOutputChannel('Vortex');
    const diagnostics = vscode.languages.createDiagnosticCollection('vortex');

    function runVortexCommand(subcommand: string, showOutput: boolean = true) {
        const editor = vscode.window.activeTextEditor;
        if (!editor || !editor.document.fileName.endsWith('.vx')) {
            vscode.window.showWarningMessage('No active .vx file');
            return;
        }

        const file = editor.document.fileName;
        outputChannel.clear();
        if (showOutput) {
            outputChannel.show(true);
        }

        outputChannel.appendLine(`> vortex ${subcommand} ${file}`);
        outputChannel.appendLine('');

        exec(`vortex ${subcommand} ${file}`, (err, stdout, stderr) => {
            if (stdout) {
                outputChannel.appendLine(stdout);
            }
            if (stderr) {
                outputChannel.appendLine(stderr);
            }

            if (subcommand === 'check') {
                parseDiagnostics(editor.document, stderr || stdout, diagnostics);
            }

            if (err && subcommand !== 'check') {
                vscode.window.showErrorMessage(`Vortex ${subcommand} failed: ${err.message}`);
            }
        });
    }

    const runCmd = vscode.commands.registerCommand('vortex.run', () => {
        runVortexCommand('run');
    });

    const checkCmd = vscode.commands.registerCommand('vortex.check', () => {
        runVortexCommand('check');
    });

    const codegenCmd = vscode.commands.registerCommand('vortex.codegen', () => {
        runVortexCommand('codegen');
    });

    context.subscriptions.push(runCmd, checkCmd, codegenCmd, diagnostics);
}

function parseDiagnostics(
    document: vscode.TextDocument,
    output: string,
    diagnostics: vscode.DiagnosticCollection
) {
    const diags: vscode.Diagnostic[] = [];
    // Match patterns like "error[E001]: message\n  --> file:line:col"
    const pattern = /(error|warning)\[?\w*\]?:\s*(.+?)(?:\n\s*-->\s*\S+:(\d+):(\d+))?/g;
    let match;

    while ((match = pattern.exec(output)) !== null) {
        const severity = match[1] === 'error'
            ? vscode.DiagnosticSeverity.Error
            : vscode.DiagnosticSeverity.Warning;
        const message = match[2].trim();
        const line = match[3] ? parseInt(match[3]) - 1 : 0;
        const col = match[4] ? parseInt(match[4]) - 1 : 0;

        const range = new vscode.Range(line, col, line, col + 1);
        diags.push(new vscode.Diagnostic(range, message, severity));
    }

    diagnostics.set(document.uri, diags);
}

export function deactivate() {}
