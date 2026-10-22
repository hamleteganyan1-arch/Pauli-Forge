import argparse
from .training import train_directory

def main():
    p = argparse.ArgumentParser(description="PauliForge research prototype")
    sub = p.add_subparsers(dest="command", required=True)
    t=sub.add_parser("train", help="train from reference .npz labels")
    t.add_argument("data_dir")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--output", default="pauliforge.pt")
    t.add_argument("--domain", default="configs/frozen_de.example.json")
    a=p.parse_args()
    if a.command == "train":
        train_directory(a.data_dir, epochs=a.epochs, output=a.output, domain_json=a.domain)

if __name__ == "__main__":
    main()
