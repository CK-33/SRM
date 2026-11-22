using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

class Launcher {
    [STAThread]
    static void Main(string[] args) {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        bool smoke = Array.IndexOf(args, "--smoke") >= 0;
        string python = Environment.GetEnvironmentVariable("SRM_LAB_PYTHON");
        if (String.IsNullOrWhiteSpace(python)) {
            string local = Path.Combine(root, ".venv", "Scripts", "pythonw.exe");
            string research = @"C:\Windows\System32\work\g5-cpu\Scripts\pythonw.exe";
            python = File.Exists(local) ? local : research;
        }
        try {
            if (!File.Exists(python))
                throw new FileNotFoundException("Configure o Python conforme o guia ou a variável SRM_LAB_PYTHON.", python);
            string demo = Array.IndexOf(args, "--demo") >= 0 ? " --demo" : "";
            if (smoke) {
                demo = " --smoke";
                if (String.Equals(Path.GetFileName(python), "pythonw.exe", StringComparison.OrdinalIgnoreCase))
                    python = Path.Combine(Path.GetDirectoryName(python), "python.exe");
            }
            var settings = new ProcessStartInfo(python, "-B \"" + Path.Combine(root, "app.py") + "\"" + demo);
            settings.WorkingDirectory = root;
            settings.UseShellExecute = false;
            settings.CreateNoWindow = true;
            var process = Process.Start(settings);
            if (smoke) {
                Environment.ExitCode = process.WaitForExit(20000) ? process.ExitCode : 2;
            }
        } catch (Exception error) {
            if (smoke) {
                File.WriteAllText(Path.Combine(root, "launcher-smoke-erro.txt"), error.ToString());
                Environment.ExitCode = 1;
            } else {
                MessageBox.Show(error.Message, "SRM Lab", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
    }
}
