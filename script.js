document.addEventListener("DOMContentLoaded", function() {
    var genreOptions = document.getElementById("genreOptions");

    // Define an array of genres
    var genres = ["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"];

    // Populate dropdown menu with checkboxes
    genres.forEach(function(genre) {
        var checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.id = genre.toLowerCase();
        checkbox.name = "genre";
        checkbox.value = genre;

        var label = document.createElement("label");
        label.htmlFor = genre.toLowerCase();
        label.textContent = genre;

        genreOptions.appendChild(checkbox);
        genreOptions.appendChild(label);
        genreOptions.appendChild(document.createElement("br"));
    });

    document.getElementById('movieForm').addEventListener('submit', function(event) {
        event.preventDefault();

        // Get form values
        var movieName = document.getElementById('movieName').value;
        var description = document.getElementById('description').value;
        var actor = document.getElementById('actor').value;
        var director = document.getElementById('director').value;
        var rank = document.getElementById('rank').value;
        var year = document.getElementById('year').value;
        var runtime = document.getElementById('runtime').value;
        var rating = document.getElementById('rating').value;
        var revenue = document.getElementById('revenue').value;
        var metascore = document.getElementById('metascore').value;
        var votesTransformed = document.getElementById('votesTransformed').value;

        // Get selected genres
        var genres = [];
        var checkboxes = document.querySelectorAll('input[name="genre"]:checked');
        checkboxes.forEach(function(checkbox) {
            genres.push(checkbox.value);
        });

        // Send data to backend
        fetch('http://localhost:8501/store_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                movieName: movieName,
                description: description,
                actor: actor,
                director: director,
                rank: rank,
                year: year,
                runtime: runtime,
                rating: rating,
                revenue: revenue,
                metascore: metascore,
                votesTransformed: votesTransformed,
                genres: genres
            })
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('message').innerText = data.message;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});
