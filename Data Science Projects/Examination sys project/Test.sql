use [Examination] 

-- add few branches

exec dbo.disp_branch_proc -- displays the branches before adding

exec dbo.add_branch_proc 'Smart Village' -- adds some branches
exec dbo.add_branch_proc 'Assuit'
exec dbo.add_branch_proc 'Ismailia'
exec dbo.add_branch_proc 'Menofia'
exec dbo.add_branch_proc 'Alexandria'
exec dbo.add_branch_proc 'Aswan'
exec dbo.add_branch_proc 'Mansoura'
exec dbo.add_branch_proc 'Minya'

exec dbo.disp_branch_proc -- displays after insertion

-- add intake

exec dbo.disp_intake_proc -- displays the intakes before adding

exec dbo.add_intake_proc 'one', 'Smart Village' -- adds intakes in branches
exec dbo.add_intake_proc 'two', 'Smart Village'
exec dbo.add_intake_proc 'three', 'Smart Village'
exec dbo.add_intake_proc 'four', 'Smart Village'
exec dbo.add_intake_proc 'five', 'Smart Village'

exec dbo.add_intake_proc '2020Q1', 'Assuit' -- only smart village has the 9 months here
exec dbo.add_intake_proc '2020Q2', 'Assuit'
exec dbo.add_intake_proc '2021Q1', 'Assuit'
exec dbo.add_intake_proc '2021Q2', 'Assuit'
exec dbo.add_intake_proc '2022Q1', 'Assuit'
exec dbo.add_intake_proc '2022Q2', 'Assuit'
exec dbo.add_intake_proc '2022Q2', 'Assuit'
exec dbo.add_intake_proc '2023Q1', 'Assuit'
exec dbo.add_intake_proc '2023Q2', 'Assuit'

exec dbo.add_intake_proc '2020Q1', 'Ismailia'
exec dbo.add_intake_proc '2020Q2', 'Ismailia'
exec dbo.add_intake_proc '2021Q1', 'Ismailia'
exec dbo.add_intake_proc '2021Q2', 'Ismailia'
exec dbo.add_intake_proc '2022Q1', 'Ismailia'
exec dbo.add_intake_proc '2022Q2', 'Ismailia'
exec dbo.add_intake_proc '2022Q2', 'Ismailia'
exec dbo.add_intake_proc '2023Q1', 'Ismailia'
exec dbo.add_intake_proc '2023Q2', 'Ismailia'

exec dbo.add_intake_proc '2020Q1', 'Menofia'
exec dbo.add_intake_proc '2020Q2', 'Menofia'
exec dbo.add_intake_proc '2021Q1', 'Menofia'
exec dbo.add_intake_proc '2021Q2', 'Menofia'
exec dbo.add_intake_proc '2022Q1', 'Menofia'
exec dbo.add_intake_proc '2022Q2', 'Menofia'
exec dbo.add_intake_proc '2022Q2', 'Menofia'
exec dbo.add_intake_proc '2023Q1', 'Menofia'
exec dbo.add_intake_proc '2023Q2', 'Menofia'

exec dbo.add_intake_proc '2020Q1', 'Alexandria'
exec dbo.add_intake_proc '2020Q2', 'Alexandria'
exec dbo.add_intake_proc '2021Q1', 'Alexandria'
exec dbo.add_intake_proc '2021Q2', 'Alexandria'
exec dbo.add_intake_proc '2022Q1', 'Alexandria'
exec dbo.add_intake_proc '2022Q2', 'Alexandria'
exec dbo.add_intake_proc '2022Q2', 'Alexandria'
exec dbo.add_intake_proc '2023Q1', 'Alexandria'
exec dbo.add_intake_proc '2023Q2', 'Alexandria'

exec dbo.add_intake_proc '2020Q1', 'Aswan'
exec dbo.add_intake_proc '2020Q2', 'Aswan'
exec dbo.add_intake_proc '2021Q1', 'Aswan'
exec dbo.add_intake_proc '2021Q2', 'Aswan'
exec dbo.add_intake_proc '2022Q1', 'Aswan'
exec dbo.add_intake_proc '2022Q2', 'Aswan'
exec dbo.add_intake_proc '2022Q2', 'Aswan'
exec dbo.add_intake_proc '2023Q1', 'Aswan'
exec dbo.add_intake_proc '2023Q2', 'Aswan'

exec dbo.add_intake_proc '2020Q1', 'Mansoura'
exec dbo.add_intake_proc '2020Q2', 'Mansoura'
exec dbo.add_intake_proc '2021Q1', 'Mansoura'
exec dbo.add_intake_proc '2021Q2', 'Mansoura'
exec dbo.add_intake_proc '2022Q1', 'Mansoura'
exec dbo.add_intake_proc '2022Q2', 'Mansoura'
exec dbo.add_intake_proc '2022Q2', 'Mansoura'
exec dbo.add_intake_proc '2023Q1', 'Mansoura'
exec dbo.add_intake_proc '2023Q2', 'Mansoura'

exec dbo.add_intake_proc '2020Q1', 'Minya'
exec dbo.add_intake_proc '2020Q2', 'Minya'
exec dbo.add_intake_proc '2021Q1', 'Minya'
exec dbo.add_intake_proc '2021Q2', 'Minya'
exec dbo.add_intake_proc '2022Q1', 'Minya'
exec dbo.add_intake_proc '2022Q2', 'Minya'
exec dbo.add_intake_proc '2022Q2', 'Minya'
exec dbo.add_intake_proc '2023Q1', 'Minya'
exec dbo.add_intake_proc '2023Q2', 'Minya'

exec dbo.disp_intake_proc -- displays after insertion

-- add track

exec add_track_branch_proc 'OS','one', 'Smart Village' -- add tracks in intake and branches
exec add_track_branch_proc 'SD','two', 'Smart Village'
exec add_track_branch_proc 'DS','three', 'Smart Village'
exec add_track_branch_proc 'DS','four', 'Smart Village'
exec add_track_branch_proc 'DA','five', 'Smart Village'

exec add_track_branch_proc '.net','2020Q1', 'Assuit'
exec add_track_branch_proc 'python','2020Q2', 'Assuit'

exec add_track_branch_proc '.net','2021Q1', 'Ismailia'
exec add_track_branch_proc 'python','2021Q2', 'Ismailia'

exec add_track_branch_proc '.net','2022Q1', 'Menofia'
exec add_track_branch_proc 'python','2022Q2', 'Menofia'

exec add_track_branch_proc '.net','2022Q1', 'Alexandria'

exec add_track_branch_proc '.net','2022Q1', 'Aswan'

exec add_track_branch_proc '.net','2022Q1', 'Mansoura'


exec add_track_branch_proc '.net','2022Q1', 'Minya'
exec add_track_branch_proc 'BI','2022Q2', 'Minya'

exec add_track_branch_proc 'BI','2023Q1', 'Minya'
exec add_track_branch_proc 'BI','2023Q2', 'Minya' -- will work on only the last 4

exec add_track_branch_proc 'python', '2023Q2', 'Minya'
exec add_track_branch_proc '.net', '2023Q2', 'Minya'
exec add_track_branch_proc 'web', '2023Q2', 'Minya'


---------------------------------------------------------------------------------------------


-- display all

exec dbo.disp_all_proc -- displays after insertion

exec disp_student_proc -- display before insertion

exec add_student_proc 'Rana', 'Minia Uni', '01097363250', '2023', 'BI', '2023Q2', 'Minya'
exec add_student_proc 'Mohamed', 'Assuit Uni', '01012345678', '2020', 'python', '2023Q2', 'Minya'
exec add_student_proc 'Ahmed', 'Cairo Uni', '01112345678', '2021', '.net', '2023Q2', 'Minya'
exec add_student_proc 'Moaz', 'Alexandria Uni', '01212345678', '2022', 'web', '2023Q2', 'Minya'

exec disp_student_proc -- display after insertion

-- adds dummy to delete
exec add_student_proc 'test', 'Alexandria Uni', '01212345678', '2022', 'web', '2023Q2', 'Minya'
exec disp_student_proc

exec delete_student_proc 'test', 'web', '2023Q2', 'Minya'
exec disp_student_proc -- delete successful 

----------------------------------------------------------------------------------------------


-- adds instructors
exec [dbo].[add_instructor_proc] 'ahmed'
exec [dbo].[add_instructor_proc] 'ali'
exec [dbo].[add_instructor_proc] 'khaled'
exec [dbo].[add_instructor_proc] 'abdo'
exec [dbo].[add_instructor_proc] 'saeed'
exec [dbo].[add_instructor_proc] 'mostafa'
exec [dbo].[add_instructor_proc] 'moaaz'
exec [dbo].[add_instructor_proc] 'taha'

-- displays after insertion
exec [dbo].[disp_instructor_proc]

-- edit name of instructor taha
exec [dbo].[edit_instructor_proc] 'taha' , 'rami', 8
exec [dbo].[disp_instructor_proc] -- displays the edit

exec [dbo].[delete_instructor_proc] 16, 'taha' -- delete instructor

exec [dbo].[add_instructor_proc] 'Rana' -- dummy data 
exec [dbo].[delete_instructor_proc] 9, 'Rana' -- delete dummy data
exec [dbo].[disp_instructor_proc] -- shows delete


-------------------------------------------------------------------------------------------------------

-- insert course
exec add_course_proc 'SQL', 'Basic database concepts', 0, 100, 8;
exec add_course_proc 'Python', 'Fundamentals of programming', 0, 100, 1;
exec add_course_proc 'Power BI', 'Building interactive dashboards', 0, 100, 2;
exec add_course_proc 'Data Science Fundamentals', 'Introduction to data analysis', 0, 100, 3;
exec add_course_proc 'Network Security', 'Securing computer networks', 0, 100, 8;
exec add_course_proc 'Data Mining', 'Overview of statistics and AI techniques', 0, 100, 1;
exec add_course_proc 'Web design basics', 'Overview of HTML and CSS', 0, 100,4;


exec delete_course_proc 5; -- delete course with id 5

exec edit_course_proc 'SQL' , 'fundamentals of database concepts', 0, 100, 8
-- edit SQL course

exec disp_course_proc; -- display after edits


select * from [dbo].[get_course_info_by_name_fun]('SQL'); -- get course with the name SQL

select * from dbo.get_instructor_courses_name_id_func(1, 'ahmed') 
-- get courses that ahmed teaches

--------------------------------------------------------------------------------------------
--tests if the views are showing the correct information

select * from courses_for_BI
select * from courses_for_python
select * from courses_for_dot_net
select * from courses_for_web

-----------------------------------------------------------------------------------------------------

-- test if insertion is correct
select * from view_student_courses_func()
-------------------------------------------------------------------------------------------------------------------
--- proc for making exam before assigning questions

exec make_exam_proc @ins_id = 8, @c_id = 1, @type = 0,
@year = '2023', @allownace = 'half hour extra', 
@startTime = '2023-12-1 12:00', @endTime = '2023-12-1 14:00'
-- makes exam 

declare @eid int = (select IDENT_CURRENT('Exam')) -- takes the last inserted id in the exam table
-- test make exam 
exec chose_Q_random_proc 
@exam_id = @eid, @c_id = 1, 
@NMCQ = 4, @NTF = 4, @NTXT = 4,
@MCQ_degree = 1, @TF_degree = 2, @TXT_degree = 3
-- choses 4 random question of each type and their degrees

go
exec make_exam_proc @ins_id = 1, @c_id = 2, @type = 0,
@year = '2023', @allownace = 'half hour extra', 
@startTime = '2023-12-2 12:00', @endTime = '2023-12-2 14:00'
-- make a new exam

exec disp_questions_proc -- display questions to chose from
declare @eid int = (select IDENT_CURRENT('Exam'))
exec make_exam_manual_proc @exam_id = @eid, @c_id = 2,
@Question_id_string = '1,2,3,4,5,6,7', @question_degree_string = '1,2,3,4,5,6,7'
-- choses questions and assigns degrees

select * from questionExam -- select the last two exams

---------------------------------------------------------------------------------------
exec select_students_for_exam_proc 1, 3
exec select_students_for_exam_proc 2, 3
exec select_students_for_exam_proc 4, 3
exec select_students_for_exam_proc 6, 3
-- adds students to exam

------------------------------------------------------------------------------------------
exec CheckExamAccess_proc 5,1 ;

exec CheckExamAccess_proc 4,1 ; -- checks if student is assigned this exam and displays it

-------------------------------------------------------------------------------------------

exec StoreStudentAnswer_proc 1,5,1,'True';
exec StoreStudentAnswer_proc 1,5,4,'Mars';
exec StoreStudentAnswer_proc 1,5,13,'Structured query language';
exec StoreStudentAnswer_proc 1,5,25,'True';    --False
exec StoreStudentAnswer_proc 1,5,26,'True';
exec StoreStudentAnswer_proc 1,5,28,'Ottawa';
exec StoreStudentAnswer_proc 1,5,31,'Seoul';
exec StoreStudentAnswer_proc 1,5,32,'The cheetah';
exec StoreStudentAnswer_proc 1,5,38,'False';
exec StoreStudentAnswer_proc 1,5,39,'Marie Curie';
exec StoreStudentAnswer_proc 1,5,40,'Oxygen';
exec StoreStudentAnswer_proc 1,5,41,'cairo';      --London

--------------------------------------------------------------------------------------------


exec check_answer__calc_score_proc 5, 1; -- calculates scores for exam 5 for student 1
-----------------------------------------------------------------------------------------

exec calc_final_result_for_student 5, 1;

exec calc_final_result_for_student 3, 2;

-- tests the final scores for students in exams
-----------------------------------------------------------------------------------------
	 SELECT * FROM studentcourses;
	 SELECT * FROM intake_track;
	  SELECT * FROM ins_course;