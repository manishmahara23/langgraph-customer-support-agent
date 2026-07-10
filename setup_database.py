import database


def main():
    database.init_db()
    seeded = database.seed_demo_data()

    if seeded:
        print("Database created and seeded with demo users and orders.")
    else:
        print("Database already has data - skipped seeding.")

    print("\nDemo users you can pick from in the app:")
    for user in database.list_users():
        print(f"  {user['user_id']} - {user['name']}")


if __name__ == "__main__":
    main()
