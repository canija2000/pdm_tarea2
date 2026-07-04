# Colores ANSI para terminal
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Imprime un encabezado bonito"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")

def print_step(step_num, text):
    """Imprime un paso del proceso"""
    print(f"{Colors.BOLD}{Colors.YELLOW}[Paso {step_num}]{Colors.END} {text}")

def print_success(text):
    """Imprime un mensaje de éxito"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    """Imprime un mensaje de error"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    """Imprime información"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")