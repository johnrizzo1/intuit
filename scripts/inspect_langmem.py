"""
Script to inspect the langmem package structure.
"""
import inspect
import importlib
import pkgutil

def inspect_package(package_name):
    """Inspect a package and print its structure."""
    try:
        # Import the package
        package = importlib.import_module(package_name)
        print(f"\n=== {package_name} ===")
        
        # Print package attributes
        print("\nAttributes:")
        for name in dir(package):
            if not name.startswith('_'):  # Skip private attributes
                attr = getattr(package, name)
                attr_type = type(attr).__name__
                print(f"  {name} ({attr_type})")
        
        # Print submodules
        print("\nSubmodules:")
        for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
            print(f"  {name} ({'package' if is_pkg else 'module'})")
            
            # Try to import the submodule
            try:
                submodule = importlib.import_module(name)
                
                # Print submodule attributes
                for attr_name in dir(submodule):
                    if not attr_name.startswith('_'):  # Skip private attributes
                        try:
                            attr = getattr(submodule, attr_name)
                            attr_type = type(attr).__name__
                            print(f"    {attr_name} ({attr_type})")
                        except Exception as e:
                            print(f"    {attr_name} (Error: {e})")
            except Exception as e:
                print(f"    Error importing: {e}")
                
    except ImportError as e:
        print(f"Error importing {package_name}: {e}")
    except Exception as e:
        print(f"Error inspecting {package_name}: {e}")

if __name__ == "__main__":
    inspect_package("langmem")