
create view view_Dim_status
as
with cte
as
(
select 1 status_id, [dbo].[ufnGetSalesOrderStatusText](1) [status]
Union all
select status_id+1, [dbo].[ufnGetSalesOrderStatusText](status_id+1) [status]
from cte
where status_id<6
)
select * from cte;

